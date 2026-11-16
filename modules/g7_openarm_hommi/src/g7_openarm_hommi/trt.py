"""TensorRT engine build and execution on torch CUDA tensors."""

from __future__ import annotations

from pathlib import Path

import torch


def build_engine(
    onnx_path: Path, engine_path: Path, *, fp16: bool, fp32_layers: tuple[str, ...] = ()
) -> None:
    """Equivalent to ``trtexec --onnx=<onnx> --saveEngine=<engine> [--fp16]``.

    ``fp32_layers`` names ``tensorrt.LayerType`` members that stay in fp32 inside an
    fp16 engine. TensorRT engines only run on the GPU and TensorRT version they were
    built with, so every deployment machine builds its own.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"cannot parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        pinned = {getattr(trt.LayerType, name) for name in fp32_layers}
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            if layer.type in pinned:
                layer.precision = trt.float32
                for output in range(layer.num_outputs):
                    layer.set_output_type(output, trt.float32)
        if pinned:
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    precision = "fp16" if fp16 else "fp32"
    print(f"building {engine_path} (TensorRT {trt.__version__}, {precision}) ...", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
    tmp = engine_path.with_suffix(engine_path.suffix + ".tmp")
    tmp.write_bytes(serialized)
    tmp.replace(engine_path)


class TrtEngine:
    """A static-shape float32 engine called with torch tensors on its CUDA device.

    Outputs are written into buffers owned by the engine, so a returned tensor is
    only valid until the next call.
    """

    def __init__(self, engine_path: Path, *, device: torch.device | str, build_hint: str) -> None:
        import tensorrt as trt

        if not engine_path.is_file():
            raise FileNotFoundError(
                f"no TensorRT engine at {engine_path}; build it on this machine with `{build_hint}`"
            )
        self.device = torch.device(device)
        self.path = engine_path
        self._engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
            engine_path.read_bytes()
        )
        if self._engine is None:
            raise RuntimeError(
                f"cannot load {engine_path} with TensorRT {trt.__version__}; "
                f"rebuild it with `{build_hint}`"
            )
        self._context = self._engine.create_execution_context()
        self._stream = torch.cuda.Stream(device=self.device)
        self.input_shapes: dict[str, tuple[int, ...]] = {}
        self._outputs: dict[str, torch.Tensor] = {}
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            if self._engine.get_tensor_dtype(name) != trt.DataType.FLOAT:
                raise RuntimeError(f"{engine_path}: tensor {name} is not float32")
            shape = tuple(self._engine.get_tensor_shape(name))
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_shapes[name] = shape
            else:
                self._outputs[name] = torch.empty(shape, device=self.device)
                self._context.set_tensor_address(name, self._outputs[name].data_ptr())

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.keys() != self.input_shapes.keys():
            raise ValueError(
                f"{self.path.name} takes {sorted(self.input_shapes)}, got {sorted(inputs)}"
            )
        # Keeps every converted input alive until the engine has read it.
        bound = []
        for name, value in inputs.items():
            tensor = value.to(device=self.device, dtype=torch.float32).contiguous()
            if tuple(tensor.shape) != self.input_shapes[name]:
                raise ValueError(
                    f"{self.path.name}: {name} is {tuple(tensor.shape)}, "
                    f"expected {self.input_shapes[name]}"
                )
            self._context.set_tensor_address(name, tensor.data_ptr())
            bound.append(tensor)
        # The inputs were produced on the caller's stream.
        torch.cuda.current_stream(self.device).synchronize()
        if not self._context.execute_async_v3(self._stream.cuda_stream):
            raise RuntimeError(f"TensorRT inference failed for {self.path.name}")
        self._stream.synchronize()
        return self._outputs
