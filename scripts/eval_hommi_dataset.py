"""Offline open-loop evaluation of the HoMMI ego3d policy on an iPhUMI export.

For every query frame (every ``--stride`` 60 Hz frames, HoMMI samples every 8)
the observation window is rebuilt from the recording exactly like
``UmiTaskDataset`` (see ``g7_openarm_hommi.dataset``), passed through the
deployed ``Ego3dObservationBuilder`` + policy, and the predicted 32-step chunk
is compared with the ground-truth chunk. A "hold" baseline (repeat the current
state) is reported next to it, so the numbers show how much of the motion the
policy actually predicts.

    uv run --with matplotlib python scripts/eval_hommi_dataset.py \\
        ~/Downloads/1014-demo-cropped --out logs/hommi_eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from g7_openarm_hommi.dataset import (
    FPS,
    Episode,
    demonstrations,
    discover,
    load_episode,
    obs_indices,
)
from g7_openarm_hommi.geometry import rotation_6d_to_matrix_umi
from g7_openarm_hommi.observation import Ego3dObservationBuilder, ObservationSample
from g7_openarm_hommi.policy import HommiPolicyRunner
from g7_openarm_hommi.state import RobotSnapshot
from g7_openarm_hommi.trajectory import ARM_SLICES, LOOKAT
from g7_openarm_hommi.vcam import HeadObservation

UNITS = {"pos": (100, "cm"), "rot": (1, "deg"), "width": (1000, "mm"), "lookat": (100, "cm")}


def chunk_errors(pred: np.ndarray, gt: np.ndarray) -> dict[str, np.ndarray]:
    """Per-step errors: positions in m, rotations in deg, widths in m."""
    errs = {"lookat": np.linalg.norm(pred[:, LOOKAT] - gt[:, LOOKAT], axis=-1)}
    for side, (pos, rot, w) in ARM_SLICES.items():
        errs[f"{side}_pos"] = np.linalg.norm(pred[:, pos] - gt[:, pos], axis=-1)
        rp = rotation_6d_to_matrix_umi(pred[:, rot])
        rg = rotation_6d_to_matrix_umi(gt[:, rot])
        cos = (np.einsum("nij,nij->n", rp, rg) - 1.0) / 2.0
        errs[f"{side}_rot"] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        errs[f"{side}_width"] = np.abs(pred[:, w] - gt[:, w])
    return errs


def summarize(errs: list[dict[str, np.ndarray]], n_exec: int, offset: int) -> dict:
    out = {}
    for key in errs[0]:
        stack = np.stack([e[key] for e in errs])  # [N, horizon]
        out[key] = {
            "exec_mean": float(stack[:, offset : offset + n_exec].mean()),
            "chunk_mean": float(stack.mean()),
            "final_mean": float(stack[:, -1].mean()),
        }
    return out


def format_summary(name: str, policy: dict, hold: dict) -> str:
    lines = [
        f"\n== {name} ==",
        f"{'metric':<16}{'policy exec':>12}{'policy 32':>11}{'policy @end':>12}"
        f"{'hold exec':>11}{'hold 32':>9}{'hold @end':>10}",
    ]
    for key, p in policy.items():
        scale, unit = UNITS[key.split("_")[-1]]
        h = hold[key]
        vals = [
            p["exec_mean"],
            p["chunk_mean"],
            p["final_mean"],
            h["exec_mean"],
            h["chunk_mean"],
            h["final_mean"],
        ]
        widths = (12, 11, 12, 11, 9, 10)
        lines.append(
            f"{key + ' [' + unit + ']':<16}"
            + "".join(f"{v * scale:>{w}.2f}" for v, w in zip(vals, widths, strict=True))
        )
    return "\n".join(lines)


def plot_episode(
    path: Path, ep: Episode, results: list[dict], step: int, n_exec: int, offset: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = ep.times - ep.times[0]
    lookat_gt = np.stack([ep.lookat_world(i) for i in range(ep.length)])
    fig, axes = plt.subplots(4, 3, figsize=(16, 13), sharex=True)

    def chunk_times(res: dict, n: int) -> np.ndarray:
        return t[res["cur"]] + np.arange(n) * step / FPS

    def draw(ax, res: dict, values: np.ndarray) -> None:
        tt = chunk_times(res, len(values))
        ax.plot(tt, values, color="tab:blue", lw=0.5, alpha=0.35)
        ax.plot(
            tt[offset : offset + n_exec], values[offset : offset + n_exec], color="tab:red", lw=1.0
        )

    for r, side in enumerate(("left", "right")):
        pos, rot, _ = ARM_SLICES[side]
        for c, axis in enumerate("xyz"):
            ax = axes[r, c]
            ax.plot(t, ep.tcp[side][:, c, 3], "k", lw=1.5)
            for res in results:
                base = ep.tcp["left"][res["cur"]]
                rel = np.tile(np.eye(4), (len(res["pred"]), 1, 1))
                rel[:, :3, 3] = res["pred"][:, pos]
                rel[:, :3, :3] = rotation_6d_to_matrix_umi(res["pred"][:, rot])
                draw(ax, res, (base @ rel)[:, c, 3])
            ax.set_title(f"{side} TCP {axis} [m] (world)")
    for c, side in enumerate(("left", "right")):
        ax = axes[2, c]
        ax.plot(t, ep.width[side], "k", lw=1.5)
        for res in results:
            draw(ax, res, res["pred"][:, ARM_SLICES[side][2]])
        ax.set_title(f"{side} gripper width [m]")
    axes[2, 2].plot(t, ep.lookat_z, "k", lw=1.5)
    axes[2, 2].set_title("look-at depth in head frame [m] (GT)")
    for c, axis in enumerate("xyz"):
        ax = axes[3, c]
        ax.plot(t, lookat_gt[:, c], "k", lw=1.5)
        for res in results:
            base = ep.tcp["left"][res["cur"]]
            draw(ax, res, (res["pred"][:, LOOKAT] @ base[:3, :3].T + base[:3, 3])[:, c])
        ax.set_title(f"look-at point {axis} [m] (world)")
        ax.set_xlabel("time [s]")
    fig.suptitle(
        f"{ep.name}\nblack: ground truth | red: executed predicted steps "
        "| blue: full predicted chunk"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", type=Path, help="iPhUMI export directory (searched recursively)")
    parser.add_argument(
        "--out", type=Path, required=True, help="output directory for metrics/plots"
    )
    parser.add_argument(
        "--stride", type=int, default=8, help="query every N 60 Hz frames (HoMMI: 8)"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    runner = HommiPolicyRunner(device=args.device, model_dir=args.model_dir)
    builder = Ego3dObservationBuilder(runner.shape_meta, crop=runner.crop, device=runner.device)
    meta = runner.shape_meta
    obs_step = int(meta["obs"]["camera_head_pointmap"]["down_sample_steps"])
    horizon = int(meta["action"]["horizon"])
    step = int(meta["action"]["down_sample_steps"])
    n_exec, offset = runner.n_action_steps, runner.n_obs_steps - 1
    print(
        f"model {runner.model_path} | obs {builder.horizon}x{obs_step} | action {horizon}x{step} "
        f"| executed steps {offset}..{offset + n_exec - 1} | {runner.crop}"
    )

    recordings = discover(args.root)
    demos = demonstrations(recordings)
    if not demos:
        raise SystemExit(f"no complete head/left/right demonstrations under {args.root}")
    args.out.mkdir(parents=True, exist_ok=True)
    sizes = {"head": builder.head_size, "left": builder.wrist_size, "right": builder.wrist_size}

    calibrations: dict = {}
    all_policy, all_hold, report = [], [], {"episodes": {}}
    for rec in demos:
        print(f"\n[{rec.title}]")
        ep = load_episode(
            rec,
            recordings,
            head_size=builder.head_size,
            calibrations=calibrations,
            log=lambda m: print("  " + m),
        )
        curs = list(range(0, ep.length - (horizon - 1) * step, args.stride))
        print(
            f"  {ep.length} frames ({ep.length / FPS:.2f}s), {len(curs)} queries, "
            f"tag detection rate {ep.detection_rate}"
        )
        frames = {i for cur in curs for i in obs_indices(cur, builder.horizon, obs_step)}
        rgb = ep.load_rgb(frames, sizes)

        results, pol_errs, hold_errs = [], [], []
        for cur in curs:
            samples = []
            for i in obs_indices(cur, builder.horizon, obs_step):
                robot = RobotSnapshot(
                    timestamp=float(ep.times[i]),
                    left_tcp=ep.tcp["left"][i].astype(np.float32),
                    right_tcp=ep.tcp["right"][i].astype(np.float32),
                    head=ep.tcp["head"][i].astype(np.float32),
                    left_gripper_width=float(ep.width["left"][i]),
                    right_gripper_width=float(ep.width["right"][i]),
                )
                samples.append(
                    ObservationSample(
                        sequence=i,
                        timestamp=float(ep.times[i]),
                        head=HeadObservation(
                            rgb=rgb["head"][i], pointmap=ep.pointmap(i, builder.head_size)
                        ),
                        wrist_rgb={s: rgb[s][i] for s in ("left", "right")},
                        robot=robot,
                    )
                )
            obs = builder.build(samples)
            with torch.inference_mode(), runner._autocast:
                pred = runner.policy.predict_action(obs)["action_pred"][0].float().cpu().numpy()
            gt = ep.gt_action(cur, horizon, step)
            pol_errs.append(chunk_errors(pred, gt))
            hold_errs.append(chunk_errors(np.repeat(gt[:1], horizon, axis=0), gt))
            results.append({"cur": cur, "pred": pred, "gt": gt})

        pol = summarize(pol_errs, n_exec, offset)
        hold = summarize(hold_errs, n_exec, offset)
        print(format_summary(rec.title, pol, hold))
        all_policy += pol_errs
        all_hold += hold_errs
        report["episodes"][rec.title] = {
            "frames": ep.length,
            "queries": len(curs),
            "detection_rate": ep.detection_rate,
            "policy": pol,
            "hold_baseline": hold,
        }
        np.savez_compressed(
            args.out / f"{rec.title}.npz",
            cur=np.array([r["cur"] for r in results]),
            pred=np.stack([r["pred"] for r in results]),
            gt=np.stack([r["gt"] for r in results]),
        )
        if not args.no_plot:
            plot_episode(args.out / f"{rec.title}.png", ep, results, step, n_exec, offset)

    pol = summarize(all_policy, n_exec, offset)
    hold = summarize(all_hold, n_exec, offset)
    print(format_summary(f"ALL ({len(all_policy)} queries)", pol, hold))
    report["overall"] = {"policy": pol, "hold_baseline": hold}
    report["gripper_calibrations"] = {f"{k[0]}|{k[1]}": vars(v) for k, v in calibrations.items()}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
