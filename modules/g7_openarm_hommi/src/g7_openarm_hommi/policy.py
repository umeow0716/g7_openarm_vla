from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from diffusers import DDIMScheduler
from hommi_diffusion_policy import (
    DiffusionDiTImagePolicy,
    DiTObs3DEncoder,
    DiTObs3DEncoderConfig,
)

from .checkpoint import ARTIFACT_FORMAT
from .gripper import GripperWidthRange

FloatArray = npt.NDArray[np.float32]

DEFAULT_ARTIFACT_NAME = "ego3d_policy.pt"

_POLICY_KWARGS = (
    "horizon",
    "n_action_steps",
    "n_obs_steps",
    "num_inference_steps",
    "obs_as_global_cond",
    "attention_embed_dim",
    "diffusion_timestep_embed_dim",
    "depth",
    "num_heads",
    "mlp_ratio",
    "qkv_bias",
    "use_rms_norm",
)


def module_model_dir() -> Path:
    # .../modules/g7_openarm_hommi/src/g7_openarm_hommi/policy.py
    return Path(__file__).resolve().parents[2] / "model"


def _eval_crop_ratio(encoder_cfg: dict[str, Any], rgb_keys: list[str]) -> float:
    transforms = (encoder_cfg.get("eval_image_transforms") or {}).get("transforms") or {}
    ratios = set()
    for key in rgb_keys:
        steps = transforms.get(key) or []
        crops = [s for s in steps if isinstance(s, dict) and s.get("type") == "CenterCrop"]
        if len(steps) != len(crops) or len(crops) > 1:
            raise ValueError(f"unsupported eval transforms for {key}: {steps}")
        ratios.add(float(crops[0]["ratio"]) if crops else 1.0)
    if len(ratios) != 1:
        raise ValueError(f"wrist cameras use different eval crops: {ratios}")
    return ratios.pop()


def build_ego3d_policy(cfg: dict[str, Any]) -> DiffusionDiTImagePolicy:
    model_cfg = cfg["model"]
    encoder_cfg = model_cfg["obs_encoder"]
    shape_meta = cfg["shape_meta"]
    rgb_keys = sorted(
        k
        for k, a in shape_meta["obs"].items()
        if a.get("type") == "rgb" and not a.get("ignore_by_policy", False)
    )
    if encoder_cfg.get("head_wrist_attention", "none") != "none":
        raise ValueError("head_wrist_attention != 'none' is not supported")
    if encoder_cfg.get("add_pos_emb") or encoder_cfg.get("wrist_pos_emb"):
        raise ValueError("add_pos_emb / wrist_pos_emb are not supported")
    if encoder_cfg.get("xyz_proj_type", "nerf") != "nerf":
        raise ValueError("only xyz_proj_type='nerf' is supported")

    encoder = DiTObs3DEncoder(
        shape_meta,
        config=DiTObs3DEncoderConfig(
            model_name=encoder_cfg["model_name"],
            pretrained=False,  # every backbone weight comes from the checkpoint
            feature_aggregation=encoder_cfg.get("feature_aggregation", "cls"),
            share_rgb_model=bool(encoder_cfg.get("share_rgb_model", True)),
            share_pointcloud_model=bool(encoder_cfg.get("share_pointcloud_model", True)),
            use_vision_norm=bool(encoder_cfg.get("use_vision_norm", True)),
            position_embedding_dim=int(encoder_cfg.get("position_embedding_dim", 60)),
            num_points=int(encoder_cfg.get("num_points", 512)),
            downsample_mode=encoder_cfg.get("downsample_mode", "feat"),
            n_emb=int(encoder_cfg.get("n_emb", 768)),
            eval_crop_ratio=_eval_crop_ratio(encoder_cfg, rgb_keys),
        ),
    )
    scheduler_cfg = {k: v for k, v in model_cfg["noise_scheduler"].items() if k != "_target_"}
    if not model_cfg["noise_scheduler"].get("_target_", "").endswith("DDIMScheduler"):
        raise ValueError("only DDIMScheduler checkpoints are supported")
    return DiffusionDiTImagePolicy(
        name=model_cfg.get("name"),
        shape_meta=shape_meta,
        noise_scheduler=DDIMScheduler(**scheduler_cfg),
        obs_encoder=encoder,
        **{k: model_cfg[k] for k in _POLICY_KWARGS if k in model_cfg},
    )


class HommiPolicyRunner:
    """Load the ego3d HoMMI artifact once and expose numpy action-chunk inference."""

    def __init__(self, *, device: str = "cuda:0", model_dir: str | Path | None = None) -> None:
        self.device = torch.device(device)
        directory = Path(model_dir) if model_dir is not None else module_model_dir()
        path = (directory / DEFAULT_ARTIFACT_NAME).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"No HoMMI ego3d artifact at {path}. Convert a training checkpoint with "
                "`uv run python scripts/convert_hommi_ckpt.py <latest.ckpt>`."
            )
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact.get("format") != ARTIFACT_FORMAT:
            raise ValueError(
                f"{path} has format {artifact.get('format')!r}, expected {ARTIFACT_FORMAT!r}"
            )

        self.cfg: dict[str, Any] = artifact["cfg"]
        self.shape_meta: dict[str, Any] = artifact["shape_meta"]
        policy = build_ego3d_policy(self.cfg)
        policy.load_state_dict(artifact["state_dict"], strict=True)
        self.policy = policy.to(self.device).eval().requires_grad_(False)
        self.model_path = path

        self.n_action_steps = int(self.policy.n_action_steps)
        self.n_obs_steps = int(self.policy.n_obs_steps)
        self.action_dim = int(self.policy.action_dim)
        dataset_cfg = self.cfg.get("task", {}).get("dataset", {})
        if dataset_cfg.get("reference_frame", "left") != "left":
            raise ValueError("only reference_frame='left' ego3d checkpoints are supported")
        if dataset_cfg.get("pose_repr", {}).get("action_pose_repr", "relative") != "relative":
            raise ValueError("only relative action_pose_repr is supported")
        stats = self.policy.normalizer.params_dict
        self.gripper_ranges = {
            side: GripperWidthRange(
                closed=float(stats[f"gripper_{side}_gripper_width"]["input_stats"]["min"][0]),
                opened=float(stats[f"gripper_{side}_gripper_width"]["input_stats"]["max"][0]),
            )
            for side in ("left", "right")
        }
        self.depth_range = tuple(float(x) for x in dataset_cfg.get("depth_range", (0.1, 4.0)))

        # Run inference with the precision the policy was trained with
        # (accelerate's training.mixed_precision: "no" | "bf16" | "fp16").
        self.precision = str(self.cfg.get("training", {}).get("mixed_precision", "no"))
        autocast_dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16}
        if self.precision == "no":
            self._autocast = nullcontext()
        elif self.precision in autocast_dtypes:
            self._autocast = torch.autocast(
                device_type=self.device.type, dtype=autocast_dtypes[self.precision]
            )
        else:
            raise ValueError(f"unsupported training.mixed_precision {self.precision!r}")

    def predict(self, obs: dict[str, torch.Tensor]) -> FloatArray:
        with torch.inference_mode(), self._autocast:
            prediction = self.policy.predict_action(obs)
        action = prediction["action"][0].float().cpu().numpy().astype(np.float32, copy=False)
        if action.shape != (self.n_action_steps, self.action_dim):
            raise RuntimeError(
                f"unexpected HoMMI action chunk shape {action.shape}; "
                f"expected ({self.n_action_steps}, {self.action_dim})"
            )
        if not np.all(np.isfinite(action)):
            raise RuntimeError("HoMMI action chunk contains non-finite values")
        return action
