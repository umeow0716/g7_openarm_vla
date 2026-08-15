from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from hommi_train import (
    configure_evaluation_backend,
    hommi_train_config_from_mapping,
    load_portable_policy,
    load_tensorrt_policy,
    resolve_device,
    resolve_precision,
)

FloatArray = npt.NDArray[np.float32]


def module_model_dir() -> Path:
    # .../modules/g7_openarm_hommi/src/g7_openarm_hommi/policy.py
    return Path(__file__).resolve().parents[2] / "model"


class HommiPolicyRunner:
    """Load the HoMMI artifact once and expose numpy action-chunk inference."""

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        model_dir: str | Path | None = None,
    ) -> None:
        self.device = resolve_device(device)
        if self.device.type != "cuda":
            raise RuntimeError(
                f"HoMMI TensorRT deployment requires CUDA, got {self.device}"
            )

        directory = (
            Path(model_dir).expanduser().resolve()
            if model_dir is not None
            else module_model_dir()
        )
        trt_candidates = (
            directory / "model.trt.eg",
            directory / "model.trt.ep",
        )
        trt_path = next((path for path in trt_candidates if path.is_file()), None)

        if trt_path is not None:
            self.policy, self.artifact = load_tensorrt_policy(
                trt_path,
                device=self.device,
            )
            config = hommi_train_config_from_mapping(self.artifact["config"])
            self.precision = self.artifact["tensorrt_bundle"]["precision"]
            self.backend = "tensorrt-aot"
            self.model_path = trt_path
        else:
            portable = directory / "model.pt"
            if not portable.is_file():
                expected = ", ".join(path.name for path in (*trt_candidates, portable))
                raise FileNotFoundError(
                    f"No HoMMI model artifact found in {directory}. Expected one of: {expected}"
                )

            self.policy, self.artifact = load_portable_policy(
                portable,
                device=self.device,
            )
            config = hommi_train_config_from_mapping(self.artifact["config"])
            self.precision = resolve_precision("auto", self.device)
            self.backend = configure_evaluation_backend(
                self.policy,
                backend="tensorrt",
                device=self.device,
                compile_mode=config.evaluation.compile_mode,
                tensorrt=config.evaluation.tensorrt,
                precision=self.precision,
            )
            self.model_path = portable

        self.shape_meta: dict[str, Any] = dict(self.artifact["shape_meta"])
        self.n_action_steps = int(config.model.n_action_steps)
        if self.n_action_steps != 8:
            raise ValueError(
                "this deployment expects the user's HoMMI n_action_steps=8, "
                f"model config has {self.n_action_steps}"
            )

        self._autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.precision == "bf16"
            else nullcontext()
        )

    def predict(
        self,
        obs: dict[str, torch.Tensor],
    ) -> FloatArray:
        with torch.inference_mode(), self._autocast:
            prediction = self.policy.predict_action(obs)

        action = prediction["action"][0].float().cpu().numpy().astype(
            np.float32,
            copy=False,
        )
        if action.shape != (self.n_action_steps, 10):
            raise RuntimeError(
                "unexpected HoMMI action chunk shape "
                f"{action.shape}; expected ({self.n_action_steps}, 10)"
            )
        if not np.all(np.isfinite(action)):
            raise RuntimeError("HoMMI action chunk contains non-finite values")
        return action
