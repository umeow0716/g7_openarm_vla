"""Convert HoMMI training checkpoints into a portable deployment artifact.

HoMMI ``.ckpt`` files are pickled training workspaces (OmegaConf, dill and
optimizer state). Unpickling them normally imports and executes arbitrary
classes, so this module reads them with a whitelisting unpickler: only tensor
rebuilding is executed; every other class becomes an inert record whose state
is then converted to plain Python data. The resulting artifact contains only
plain containers and tensors and loads with ``torch.load(weights_only=True)``.
"""

from __future__ import annotations

import collections
import pickle
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch._utils import _rebuild_tensor_v2

ARTIFACT_FORMAT = "g7_openarm_hommi.ego3d/1"

_STORAGE_DTYPES = {
    "FloatStorage": torch.float32,
    "DoubleStorage": torch.float64,
    "HalfStorage": torch.float16,
    "BFloat16Storage": torch.bfloat16,
    "LongStorage": torch.int64,
    "IntStorage": torch.int32,
    "ShortStorage": torch.int16,
    "CharStorage": torch.int8,
    "ByteStorage": torch.uint8,
    "BoolStorage": torch.bool,
}


class _Record:
    """Inert stand-in for any non-whitelisted pickled class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.state: Any = None
        self.items: dict[Any, Any] = {}
        self.list: list[Any] = []

    def __setstate__(self, state: Any) -> None:
        self.state = state

    def __setitem__(self, key: Any, value: Any) -> None:
        self.items[key] = value

    def append(self, value: Any) -> None:
        self.list.append(value)

    def extend(self, values: Any) -> None:
        self.list.extend(values)


class _SafeUnpickler(pickle.Unpickler):
    def __init__(self, file: Any, archive: zipfile.ZipFile, prefix: str) -> None:
        super().__init__(file)
        self._archive = archive
        self._prefix = prefix
        self._storages: dict[str, torch.TypedStorage] = {}

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor_v2
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if module == "torch" and name in _STORAGE_DTYPES:
            return _STORAGE_DTYPES[name]
        return _Record

    def persistent_load(self, pid: Any) -> Any:
        kind, dtype, key, _location, _numel = pid
        if kind != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id {kind!r}")
        if key not in self._storages:
            data = self._archive.read(f"{self._prefix}/data/{key}")
            raw = (
                torch.frombuffer(bytearray(data), dtype=torch.uint8).untyped_storage()
                if data
                else torch.empty(0, dtype=torch.uint8).untyped_storage()
            )
            self._storages[key] = torch.storage.TypedStorage(
                wrap_storage=raw, dtype=dtype, _internal=True
            )
        return self._storages[key]


def _to_plain(value: Any) -> Any:
    """Convert OmegaConf records (DictConfig/ListConfig/AnyNode) to plain data."""
    if isinstance(value, _Record):
        state = value.state
        if isinstance(state, Mapping) and "_content" in state:
            content = state["_content"]
            if isinstance(content, Mapping):
                return {k: _to_plain(v) for k, v in content.items()}
            if isinstance(content, list):
                return [_to_plain(v) for v in content]
            return content
        if isinstance(state, Mapping) and "_val" in state:
            return _to_plain(state["_val"])
        return None
    if isinstance(value, Mapping):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def read_hommi_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Return ``(cfg, ema_state_dict)`` from a HoMMI workspace checkpoint."""
    with zipfile.ZipFile(path) as archive:
        pickles = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        if len(pickles) != 1:
            raise ValueError(f"{path} is not a torch zip checkpoint")
        prefix = pickles[0][: -len("/data.pkl")]
        with archive.open(pickles[0]) as handle:
            payload = _SafeUnpickler(handle, archive, prefix).load()

    if not isinstance(payload, Mapping) or "cfg" not in payload or "state_dicts" not in payload:
        raise ValueError(f"{path} is not a HoMMI workspace checkpoint")
    cfg = _to_plain(payload["cfg"])
    state_dicts = payload["state_dicts"]
    key = "ema_model" if "ema_model" in state_dicts else "model"
    state = {k: v for k, v in state_dicts[key].items() if isinstance(v, torch.Tensor)}
    return cfg, state


def convert_checkpoint(src: str | Path, dst: str | Path) -> dict[str, Any]:
    cfg, state = read_hommi_checkpoint(src)
    model_target = cfg.get("model", {}).get("_target_", "")
    encoder_target = cfg.get("model", {}).get("obs_encoder", {}).get("_target_", "")
    if not model_target.endswith("DiffusionDiTImagePolicy") or not encoder_target.endswith(
        "DiTObs3DEncoder"
    ):
        raise ValueError(
            "only HoMMI DiffusionDiTImagePolicy + DiTObs3DEncoder checkpoints are supported, "
            f"got model={model_target!r} encoder={encoder_target!r}"
        )
    # The encoder's obs_normalizer is a duplicate reference to the policy normalizer.
    state = {k: v for k, v in state.items() if not k.startswith("obs_encoder.obs_normalizer.")}
    artifact = {
        "format": ARTIFACT_FORMAT,
        "source": str(Path(src).name),
        "cfg": cfg,
        "shape_meta": cfg["shape_meta"],
        "state_dict": state,
    }
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, dst)
    return artifact
