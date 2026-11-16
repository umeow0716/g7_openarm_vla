"""Build the Fast-FoundationStereo TensorRT engine for this machine.

    uv run python scripts/build_ffs_engine.py

Equivalent to the official ``trtexec --onnx=<onnx> --saveEngine=<engine> --fp16``.
TensorRT engines only run on the GPU and TensorRT version they were built with,
so run this once on every deployment machine (e.g. the Thor).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from g7_openarm_hommi.trt import build_engine
from g7_openarm_hommi.zed_camera import FFS_ENGINE, FFS_ONNX


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx", type=Path, default=FFS_ONNX)
    parser.add_argument("--engine", type=Path, default=FFS_ENGINE)
    args = parser.parse_args()

    build_engine(args.onnx, args.engine, fp16=True)
    print(f"wrote {args.engine}")


if __name__ == "__main__":
    main()
