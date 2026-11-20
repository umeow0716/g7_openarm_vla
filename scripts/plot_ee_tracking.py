"""Plot how well the end effector tracks its target, from a WBC tracking log.

The WBC node writes ``logs/ee-tracking/<timestamp>.csv`` whenever it is
launched by ``scripts/real_hommi.py`` or ``scripts/sim_hommi.py``. With no path,
the newest log is used.

    uv run python scripts/plot_ee_tracking.py
    uv run python scripts/plot_ee_tracking.py \\
        "logs/ee-tracking/2026-11-20 10:00:00.csv" --arm right --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LOG_DIR = Path("logs") / "ee-tracking"
AXES = ("x", "y", "z")
MAX_LAG_S = 1.0


def latest_log() -> Path:
    logs = sorted(LOG_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise SystemExit(f"no tracking logs in {LOG_DIR}/ (run a hommi launcher first)")
    return logs[-1]


def load(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size < 2:
        raise SystemExit(f"{path} has fewer than 2 rows")
    return {name: data[name] for name in data.dtype.names}


def pose(d: dict[str, np.ndarray], arm: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    prefix = f"{arm}_{kind}_"
    pos = np.stack([d[prefix + a] for a in AXES], axis=1)
    quat = np.stack([d[prefix + f] for f in ("qw", "qx", "qy", "qz")], axis=1)
    return pos, quat


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=1, keepdims=True)
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def estimate_lag(t: np.ndarray, target: np.ndarray, ee: np.ndarray) -> float:
    """Delay (s) that best aligns the EE path with the target path (min RMS)."""
    dt = float(np.median(np.diff(t)))
    if dt <= 0.0:
        return 0.0
    max_shift = min(int(MAX_LAG_S / dt), len(t) // 2)
    best_shift, best_rms = 0, np.inf
    for shift in range(max_shift + 1):
        err = ee[shift:] - target[: len(t) - shift]
        rms = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
        if rms < best_rms:
            best_shift, best_rms = shift, rms
    return best_shift * dt


def summarize(arm: str, t, tgt_pos, ee_pos, tgt_quat, ee_quat) -> tuple[np.ndarray, np.ndarray]:
    pos_err_mm = np.linalg.norm(tgt_pos - ee_pos, axis=1) * 1000.0
    ori_err_deg = quat_angle_deg(tgt_quat, ee_quat)
    axis_rmse_mm = np.sqrt(np.mean((tgt_pos - ee_pos) ** 2, axis=0)) * 1000.0
    lag = estimate_lag(t, tgt_pos, ee_pos)
    print(f"[{arm}]  {len(t)} samples, {t[-1] - t[0]:.1f} s")
    print(
        f"  position error: rmse {np.sqrt(np.mean(pos_err_mm**2)):.1f} mm, "
        f"mean {pos_err_mm.mean():.1f} mm, p95 {np.percentile(pos_err_mm, 95):.1f} mm, "
        f"max {pos_err_mm.max():.1f} mm"
    )
    per_axis = ", ".join(f"{a} {v:.1f} mm" for a, v in zip(AXES, axis_rmse_mm, strict=True))
    print(f"  per-axis rmse:  {per_axis}")
    print(
        f"  orientation error: mean {ori_err_deg.mean():.2f} deg, "
        f"p95 {np.percentile(ori_err_deg, 95):.2f} deg, max {ori_err_deg.max():.2f} deg"
    )
    print(f"  estimated lag: {lag * 1000:.0f} ms")
    return pos_err_mm, ori_err_deg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", nargs="?", type=Path, help="tracking CSV (default: newest)")
    parser.add_argument("--arm", choices=("left", "right", "both"), default="both")
    parser.add_argument("--out", type=Path, help="output PNG (default: next to the CSV)")
    parser.add_argument("--show", action="store_true", help="also open an interactive window")
    args = parser.parse_args()

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = args.log or latest_log()
    d = load(path)
    t = d["t"] - d["t"][0]
    arms = ("left", "right") if args.arm == "both" else (args.arm,)

    print(f"log: {path}")
    fig, axes = plt.subplots(5, len(arms), sharex=True, squeeze=False, figsize=(7 * len(arms), 12))
    for col, arm in enumerate(arms):
        tgt_pos, tgt_quat = pose(d, arm, "target")
        ee_pos, ee_quat = pose(d, arm, "ee")
        pos_err_mm, ori_err_deg = summarize(arm, t, tgt_pos, ee_pos, tgt_quat, ee_quat)

        for row, a in enumerate(AXES):
            ax = axes[row, col]
            ax.plot(t, tgt_pos[:, row], "--", color="#888888", lw=1.2, label="target")
            ax.plot(t, ee_pos[:, row], color="#2a6fdb", lw=1.2, label="ee")
            ax.set_ylabel(f"{a} [m]")
            ax.grid(alpha=0.3)
        axes[0, col].set_title(f"{arm} arm")
        axes[0, col].legend(loc="upper right", frameon=False)

        ax = axes[3, col]
        ax.plot(t, pos_err_mm, color="#d1495b", lw=1.0)
        ax.set_ylabel("|pos err| [mm]")
        ax.grid(alpha=0.3)

        ax = axes[4, col]
        ax.plot(t, ori_err_deg, color="#e09f3e", lw=1.0)
        ax.set_ylabel("ori err [deg]")
        ax.set_xlabel("time [s]")
        ax.grid(alpha=0.3)

    fig.suptitle(path.name)
    fig.tight_layout()
    out = args.out or path.with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"saved: {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
