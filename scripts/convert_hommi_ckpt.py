"""Convert a HoMMI ego3d training checkpoint into the deployment artifact.

    uv run python scripts/convert_hommi_ckpt.py ~/Downloads/latest.ckpt
    uv run python scripts/convert_hommi_ckpt.py latest.ckpt -o /path/to/ego3d_policy.pt

The checkpoint is read with a whitelisting unpickler (no arbitrary code runs);
the output keeps only the config, shape_meta and EMA weights and loads with
``torch.load(weights_only=True)``. The rebuilt policy is loaded once with
``strict=True`` to verify the artifact before it is written.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from g7_openarm_hommi.checkpoint import convert_checkpoint
from g7_openarm_hommi.policy import DEFAULT_ARTIFACT_NAME, build_ego3d_policy, module_model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="HoMMI workspace .ckpt")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=module_model_dir() / DEFAULT_ARTIFACT_NAME,
        help="artifact path (default: modules/g7_openarm_hommi/model/ego3d_policy.pt)",
    )
    args = parser.parse_args()

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    artifact = convert_checkpoint(args.checkpoint.expanduser(), tmp)
    policy = build_ego3d_policy(artifact["cfg"])
    policy.load_state_dict(artifact["state_dict"], strict=True)
    tmp.replace(args.output)

    cfg = artifact["cfg"]
    params = sum(p.numel() for p in policy.parameters())
    print(f"wrote {args.output}")
    print(f"  task={cfg.get('task', {}).get('name')} source={artifact['source']}")
    print(
        f"  action_dim={policy.action_dim} horizon={policy.horizon} "
        f"n_action_steps={policy.n_action_steps} n_obs_steps={policy.n_obs_steps} "
        f"params={params / 1e6:.1f}M"
    )


if __name__ == "__main__":
    main()
