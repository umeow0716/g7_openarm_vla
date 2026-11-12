"""Build the Fast-FoundationStereo TensorRT engine for this machine.

    uv run python scripts/build_ffs_engine.py

Equivalent to the official ``trtexec --onnx=<onnx> --saveEngine=<engine> --fp16``.
TensorRT engines only run on the GPU and TensorRT version they were built with,
so run this once on every deployment machine (e.g. the Thor).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt as trt

from g7_openarm_hommi.zed_camera import FFS_ENGINE, FFS_ONNX


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx", type=Path, default=FFS_ONNX)
    parser.add_argument("--engine", type=Path, default=FFS_ENGINE)
    args = parser.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser_onnx = trt.OnnxParser(network, logger)
    if not parser_onnx.parse_from_file(str(args.onnx)):
        errors = "\n".join(str(parser_onnx.get_error(i)) for i in range(parser_onnx.num_errors))
        raise SystemExit(f"cannot parse {args.onnx}:\n{errors}")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    print(f"building {args.engine} from {args.onnx} (TensorRT {trt.__version__}, fp16) ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("TensorRT engine build failed")
    tmp = args.engine.with_suffix(".engine.tmp")
    tmp.write_bytes(serialized)
    tmp.replace(args.engine)
    print(f"wrote {args.engine}")


if __name__ == "__main__":
    main()
