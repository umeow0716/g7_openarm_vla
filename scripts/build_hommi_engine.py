"""Build the HoMMI policy TensorRT engines for this machine.

    uv run python scripts/build_hommi_engine.py

Exports the DINOv3 backbone and the DiT denoiser of ``ego3d_policy.pt`` to ONNX and
builds an engine for each, then compares the engines against the PyTorch policy.
TensorRT engines only run on the GPU and TensorRT version they were built with, so
run this once on every deployment machine, and again after converting a new checkpoint.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from g7_openarm_hommi.config import config
from g7_openarm_hommi.policy import HommiPolicyRunner, build_engines, module_model_dir


def random_obs(runner: HommiPolicyRunner, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=runner.device).manual_seed(seed)
    obs = runner.dummy_obs()
    for key, value in obs.items():
        if key.endswith("_mask"):
            obs[key] = (
                torch.rand(value.shape, generator=generator, device=runner.device) > 0.3
            ).float()
        elif key.endswith("_rgb"):
            obs[key] = torch.rand(value.shape, generator=generator, device=runner.device)
        else:
            obs[key] = 0.2 * torch.randn(value.shape, generator=generator, device=runner.device)
    return obs


def predict(runner: HommiPolicyRunner, observations: list[dict[str, torch.Tensor]]) -> np.ndarray:
    # Both backends must start every DDIM chain from the same noise.
    torch.manual_seed(0)
    return np.stack([runner.predict_horizon(obs) for obs in observations])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=module_model_dir())
    parser.add_argument("--precision", choices=("fp32", "fp16"), default=config.inference_precision)
    parser.add_argument("--device", default=config.device)
    args = parser.parse_args()

    build_engines(args.model_dir, args.precision)

    reference = HommiPolicyRunner(
        device=args.device, model_dir=args.model_dir, inference_precision="fp32", backend="torch"
    )
    observations = [random_obs(reference, seed) for seed in range(4)]
    expected = predict(reference, observations)
    del reference
    torch.cuda.empty_cache()

    runner = HommiPolicyRunner(
        device=args.device, model_dir=args.model_dir, inference_precision=args.precision
    )
    runner.warmup()
    actual = predict(runner, observations)
    start = time.perf_counter()
    for obs in observations:
        runner.predict(obs)
    elapsed_ms = (time.perf_counter() - start) / len(observations) * 1e3
    error = np.abs(actual - expected)
    print(
        f"TensorRT {args.precision} vs PyTorch fp32 on random observations: predicted actions "
        f"differ by {error.mean():.5f} on average, {error.max():.5f} at most "
        f"(mean |action| {np.abs(expected).mean():.3f}); {elapsed_ms:.0f} ms per inference"
    )
    if not np.isfinite(actual).all():
        raise SystemExit("the TensorRT policy produced non-finite actions")


if __name__ == "__main__":
    main()
