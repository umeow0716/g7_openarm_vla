from __future__ import annotations

import hashlib
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from hommi_diffusion_policy import (
    DiffusionDiTImagePolicy,
    DiTObs3DEncoder,
    DiTObs3DEncoderConfig,
)

from .checkpoint import ARTIFACT_FORMAT
from .crop import PointCloudCrop
from .gripper import GripperWidthRange
from .trt import TrtEngine, build_engine

FloatArray = npt.NDArray[np.float32]

DEFAULT_ARTIFACT_NAME = "ego3d_policy.pt"
ENGINE_BUILD_HINT = "uv run python scripts/build_hommi_engine.py"

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


class VisionStage(torch.nn.Module):
    """The shared DINOv3 backbone: wrist CLS tokens and head patch tokens.

    Takes the normalized observations: ``wrist`` is every wrist image stacked camera
    by camera, ``head`` the RGB paired with the pointmap.
    """

    def __init__(self, encoder: DiTObs3DEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, wrist: torch.Tensor, head: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder = self.encoder
        backbone = encoder.key_model_map["rgb"]
        size = encoder.wrist_image_size
        # The encoder's eval transform, written out because torchvision's version does not
        # trace: CenterCrop, then Resize(antialias=True), which only ever upsamples here and
        # antialiasing is a no-op when upsampling.
        crop = int(size * encoder.config.eval_crop_ratio)
        top = round((size - crop) / 2.0)
        wrist = wrist[..., top : top + crop, top : top + crop]
        wrist = F.interpolate(wrist, (size, size), mode="bilinear", align_corners=False)
        cls = backbone(pixel_values=encoder._normalize_image(wrist)).last_hidden_state[:, 0]
        tokens = backbone(pixel_values=encoder._normalize_image(head)).last_hidden_state
        return cls, tokens[:, 1 + encoder.num_register_tokens :]


class DenoiseStage(torch.nn.Module):
    """One DDIM noise prediction; ``timestep`` is float so the engine has no integer input."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, obs_embed: torch.Tensor, actions: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        return self.model(obs_embed, actions, timestep)


class _Stages(Protocol):
    def vision(self, wrist: torch.Tensor, head: torch.Tensor) -> tuple[torch.Tensor, ...]: ...

    def denoise(
        self, obs_embed: torch.Tensor, actions: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor: ...


class _TorchStages:
    def __init__(self, policy: DiffusionDiTImagePolicy, autocast: Any) -> None:
        self._vision = VisionStage(_encoder(policy))
        self._denoise = DenoiseStage(policy.model)
        self._autocast = autocast

    def vision(self, wrist: torch.Tensor, head: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with self._autocast:
            return self._vision(wrist, head)

    def denoise(
        self, obs_embed: torch.Tensor, actions: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        with self._autocast:
            return self._denoise(obs_embed, actions, timestep)


class _TrtStages:
    def __init__(self, model_dir: Path, precision: str, device: torch.device) -> None:
        self._vision = TrtEngine(
            engine_path(model_dir, "vision", precision), device=device, build_hint=ENGINE_BUILD_HINT
        )
        self._denoise = TrtEngine(
            engine_path(model_dir, "denoise", precision),
            device=device,
            build_hint=ENGINE_BUILD_HINT,
        )

    def vision(self, wrist: torch.Tensor, head: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self._vision(wrist=wrist, head=head)
        return out["wrist_cls"], out["head_tokens"]

    def denoise(
        self, obs_embed: torch.Tensor, actions: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        return self._denoise(obs_embed=obs_embed, actions=actions, timestep=timestep)["noise"]


def _encoder(policy: DiffusionDiTImagePolicy) -> DiTObs3DEncoder:
    return cast(DiTObs3DEncoder, policy.obs_encoder)


def onnx_path(model_dir: Path, stage: str) -> Path:
    return model_dir / f"ego3d_{stage}.onnx"


def engine_path(model_dir: Path, stage: str, precision: str) -> Path:
    return model_dir / f"ego3d_{stage}.{precision}.engine"


def _manifest_path(model_dir: Path, precision: str) -> Path:
    return model_dir / f"ego3d_engines.{precision}.json"


def _artifact_digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def farthest_point_indices(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Indices chosen by ``hommi_diffusion_policy``'s ``farthest_point_sample`` from index 0.

    That loop runs ``num_samples`` rounds of small GPU kernels. Here the GPU computes
    every pairwise distance once and the rounds run on the CPU over one row each.
    """
    batch, count, _ = points.shape
    if num_samples > count:
        raise ValueError(f"cannot sample {num_samples} points from {count}")
    points = points.float()
    distances = (points[:, :, None] - points[:, None]).square().sum(dim=-1).cpu().numpy()
    indices = np.empty((batch, num_samples), dtype=np.int64)
    for b in range(batch):
        rows = distances[b]
        nearest = np.full(count, np.inf, dtype=np.float32)
        current = 0
        for i in range(num_samples):
            indices[b, i] = current
            np.minimum(nearest, rows[current], out=nearest)
            current = int(nearest.argmax())
    return torch.from_numpy(indices).to(points.device)


def pool_pointmap(
    encoder: DiTObs3DEncoder,
    tokens: torch.Tensor,
    pointmap: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """``DiTObs3DEncoder._encode_pointmap`` from the backbone's patch tokens onwards."""
    batch = tokens.shape[0]
    side = math.isqrt(tokens.shape[1])
    if side * side != tokens.shape[1]:
        raise RuntimeError(f"expected a square patch grid, got {tokens.shape[1]} tokens")
    if mask is None:
        mask = torch.ones(batch, 1, *pointmap.shape[2:], device=pointmap.device)
    points = F.interpolate(pointmap, (side, side), mode="nearest").flatten(2).transpose(1, 2)
    mask = F.interpolate(mask.to(points.dtype), (side, side), mode="nearest")
    mask = mask.flatten(2).transpose(1, 2)

    points = points * mask
    tokens = tokens * mask
    mask = mask.squeeze(-1)

    space = tokens[..., :30] if encoder.config.downsample_mode == "feat" else points
    indices = farthest_point_indices(space, encoder.config.num_points)
    points = torch.gather(points, 1, indices.unsqueeze(-1).expand(-1, -1, points.shape[-1]))
    tokens = torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
    mask = torch.gather(mask, 1, indices)

    cloud = torch.cat([encoder.xyz_proj(points), tokens], dim=-1)
    pooled = encoder.pc_projection_map(cloud.unsqueeze(1), mask=mask.unsqueeze(1))
    return pooled.reshape(batch, -1)


def load_artifact(model_dir: Path) -> tuple[Path, dict[str, Any], DiffusionDiTImagePolicy]:
    """Return the artifact path, its contents and the fp32 policy on the CPU."""
    path = (model_dir / DEFAULT_ARTIFACT_NAME).expanduser().resolve()
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
    policy = build_ego3d_policy(artifact["cfg"])
    policy.load_state_dict(artifact["state_dict"], strict=True)
    return path, artifact, policy.eval().requires_grad_(False)


def build_engines(model_dir: Path, precision: str) -> None:
    """Export both stages to ONNX and build this machine's engines from them."""
    path, artifact, policy = load_artifact(model_dir)
    encoder = _encoder(policy)
    obs_meta = artifact["shape_meta"]["obs"]
    steps = policy.n_obs_steps
    head_key = encoder.paired_rgb_key(encoder.pointmap_keys[0])
    wrist = torch.zeros(len(encoder.rgb_keys) * steps, *obs_meta[encoder.rgb_keys[0]]["shape"])
    head = torch.zeros(steps, *obs_meta[head_key]["shape"])
    obs_embed = torch.zeros(1, policy.obs_feature_dim * steps)
    actions = torch.zeros(1, policy.horizon, policy.action_dim)
    # On some GPUs (an RTX 4060, not the Thor) an fp16 attention softmax in the backbone
    # costs the patch tokens about 20 % relative error on recorded observations; pinned
    # to fp32 the error is under 1 %.
    fp32_layers = {"vision": ("SOFTMAX",), "denoise": ()}
    exports: tuple[tuple[str, torch.nn.Module, tuple[torch.Tensor, ...], dict[str, Any]], ...]
    exports = (
        (
            "vision",
            VisionStage(encoder),
            (wrist, head),
            {"input_names": ["wrist", "head"], "output_names": ["wrist_cls", "head_tokens"]},
        ),
        (
            "denoise",
            DenoiseStage(policy.model),
            (obs_embed, actions, torch.zeros(1)),
            {"input_names": ["obs_embed", "actions", "timestep"], "output_names": ["noise"]},
        ),
    )
    for stage, module, args, names in exports:
        onnx = onnx_path(model_dir, stage)
        torch.onnx.export(module, args, str(onnx), opset_version=17, dynamo=False, **names)
        build_engine(
            onnx,
            engine_path(model_dir, stage, precision),
            fp16=precision == "fp16",
            fp32_layers=fp32_layers[stage],
        )
    _manifest_path(model_dir, precision).write_text(
        json.dumps({"artifact_sha256": _artifact_digest(path)}) + "\n"
    )


class HommiPolicyRunner:
    """Load the ego3d HoMMI artifact once and expose numpy action-chunk inference.

    The two heavy networks (DINOv3 backbone, DiT denoiser) run as stages, either
    TensorRT engines or the PyTorch modules; normalization, point pooling and the
    DDIM loop are shared by both backends.
    """

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        model_dir: str | Path | None = None,
        inference_precision: str = "fp16",
        backend: str = "tensorrt",
    ) -> None:
        if inference_precision not in ("fp32", "fp16"):
            raise ValueError(f"unsupported inference_precision {inference_precision!r}")
        if backend not in ("tensorrt", "torch"):
            raise ValueError(f"unsupported backend {backend!r}")
        self.device = torch.device(device)
        self.model_dir = Path(model_dir) if model_dir is not None else module_model_dir()
        path, artifact, policy = load_artifact(self.model_dir)
        self.cfg: dict[str, Any] = artifact["cfg"]
        self.shape_meta: dict[str, Any] = artifact["shape_meta"]
        self.policy = policy
        self.model_path = path
        self.backend = backend
        self.inference_precision = inference_precision

        self.n_action_steps = int(self.policy.n_action_steps)
        self.n_obs_steps = int(self.policy.n_obs_steps)
        self.action_dim = int(self.policy.action_dim)
        dataset_cfg = self.cfg.get("task", {}).get("dataset", {})
        if dataset_cfg.get("reference_frame", "left") != "left":
            raise ValueError("only reference_frame='left' ego3d checkpoints are supported")
        if dataset_cfg.get("pose_repr", {}).get("action_pose_repr", "relative") != "relative":
            raise ValueError("only relative action_pose_repr is supported")
        if self.policy.use_flow_matching:
            raise ValueError("flow-matching checkpoints are not supported")
        stats = self.policy.normalizer.params_dict
        self.gripper_ranges = {
            side: GripperWidthRange(
                closed=float(stats[f"gripper_{side}_gripper_width"]["input_stats"]["min"][0]),
                opened=float(stats[f"gripper_{side}_gripper_width"]["input_stats"]["max"][0]),
            )
            for side in ("left", "right")
        }
        # HoMMI UmiTaskDataset defaults: crop_pointcloud=False, gripper_length=0.13.
        depth_range = dataset_cfg.get("depth_range", (0.1, 0.8))
        self.crop = PointCloudCrop(
            enabled=bool(dataset_cfg.get("crop_pointcloud", False)),
            depth_range=(float(depth_range[0]), float(depth_range[1])),
            gripper_length=float(dataset_cfg.get("gripper_length", 0.13) or 0.13),
        )
        # accelerate's training.mixed_precision: "no" | "bf16" | "fp16".
        self.precision = str(self.cfg.get("training", {}).get("mixed_precision", "no"))

        self._stages: _Stages
        if backend == "tensorrt":
            self._check_engines()
            # Only the glue around the engines runs in PyTorch.
            self.policy.normalizer.to(self.device)
            _encoder(self.policy).pc_projection_map.to(self.device)
            self._stages = _TrtStages(self.model_dir, inference_precision, self.device)
        else:
            self.policy.to(self.device)
            self._stages = _TorchStages(self.policy, self._torch_autocast())

    def _torch_autocast(self) -> Any:
        if self.inference_precision == "fp16":
            # The denoiser (read once per DDIM step) is stored in half precision, which
            # halves its memory traffic; the stages run under fp16 autocast.
            self.policy.model.half()
            return torch.autocast(device_type=self.device.type, dtype=torch.float16)
        # Otherwise run with the precision the policy was trained with.
        autocast_dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16}
        if self.precision == "no":
            return nullcontext()
        if self.precision not in autocast_dtypes:
            raise ValueError(f"unsupported training.mixed_precision {self.precision!r}")
        return torch.autocast(device_type=self.device.type, dtype=autocast_dtypes[self.precision])

    def _check_engines(self) -> None:
        """Refuse engines built from another checkpoint: they would run silently."""
        manifest = _manifest_path(self.model_dir, self.inference_precision)
        if not manifest.is_file():
            raise FileNotFoundError(
                f"no {self.inference_precision} TensorRT engines in {self.model_dir}; "
                f"build them on this machine with `{ENGINE_BUILD_HINT}`"
            )
        if json.loads(manifest.read_text())["artifact_sha256"] != _artifact_digest(self.model_path):
            raise RuntimeError(
                f"the TensorRT engines in {self.model_dir} were built from another "
                f"{self.model_path.name}; rebuild them with `{ENGINE_BUILD_HINT}`"
            )

    def dummy_obs(self) -> dict[str, torch.Tensor]:
        """All-zero observation batch with the policy's input keys and shapes."""
        obs_meta = self.shape_meta["obs"]
        horizon = self.n_obs_steps
        obs: dict[str, torch.Tensor] = {}
        for key, attr in obs_meta.items():
            needed = not attr.get("ignore_by_policy", False) or key == "camera_head_main_rgb"
            if needed:
                obs[key] = torch.zeros((1, horizon, *attr["shape"]), device=self.device)
            if attr.get("type") == "pointmap":
                size = tuple(attr["shape"][1:])
                obs[f"{key}_mask"] = torch.ones((1, horizon, 1, *size), device=self.device)
        return obs

    def warmup(self, iterations: int = 2) -> None:
        """Run dummy inferences so the first real chunk is not slowed by CUDA warm-up."""
        obs = self.dummy_obs()
        for _ in range(iterations):
            self.predict(obs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict_horizon(self, obs: dict[str, torch.Tensor]) -> FloatArray:
        """The whole predicted action sequence ``[horizon, action_dim]`` for one observation."""
        policy = self.policy
        encoder = _encoder(policy)
        steps = self.n_obs_steps
        nobs = {
            key: value[0, :steps].float() for key, value in policy.normalizer.normalize(obs).items()
        }
        pointmap_key = encoder.pointmap_keys[0]
        wrist = torch.cat([nobs[key] for key in encoder.rgb_keys], dim=0)
        cls, tokens = self._stages.vision(wrist, nobs[encoder.paired_rgb_key(pointmap_key)])
        # [cameras, steps, D] -> one row per step, cameras side by side.
        wrist_features = cls.float().reshape(len(encoder.rgb_keys), steps, -1).transpose(0, 1)
        features = [
            wrist_features.reshape(steps, -1),
            pool_pointmap(
                encoder, tokens.float(), nobs[pointmap_key], nobs.get(f"{pointmap_key}_mask")
            ),
            *(nobs[key] for key in encoder.low_dim_keys),
        ]
        obs_embed = torch.cat(features, dim=-1).reshape(1, -1)

        scheduler: Any = policy.noise_scheduler
        scheduler.set_timesteps(policy.num_inference_steps, device=self.device)
        trajectory = torch.randn((1, policy.horizon, policy.action_dim), device=self.device)
        for timestep in scheduler.timesteps:
            noise = self._stages.denoise(obs_embed, trajectory, timestep.float().reshape(1))
            trajectory = scheduler.step(noise.float(), timestep, trajectory).prev_sample
        action = policy.normalizer["action"].unnormalize(trajectory)[0]
        return action.cpu().numpy().astype(np.float32, copy=False)

    def predict(self, obs: dict[str, torch.Tensor]) -> FloatArray:
        """The executed action chunk ``[n_action_steps, action_dim]``."""
        start = self.n_obs_steps - 1
        action = self.predict_horizon(obs)[start : start + self.n_action_steps]
        if action.shape != (self.n_action_steps, self.action_dim):
            raise RuntimeError(
                f"unexpected HoMMI action chunk shape {action.shape}; "
                f"expected ({self.n_action_steps}, {self.action_dim})"
            )
        if not np.all(np.isfinite(action)):
            raise RuntimeError("HoMMI action chunk contains non-finite values")
        return action
