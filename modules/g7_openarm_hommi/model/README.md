Place the HoMMI ego3d deployment artifact here:

- `ego3d_policy.pt` -- produced by `uv run python scripts/convert_hommi_ckpt.py <latest.ckpt>`
- `ego3d_vision.*` / `ego3d_denoise.*` / `ego3d_engines.*.json` -- the policy's ONNX exports
  and TensorRT engines, built from `ego3d_policy.pt` on each deployment machine with
  `uv run python scripts/build_hommi_engine.py`
- `ffs/20_30_48_iters_8_res_576x960.onnx` and `.yaml` -- official Fast-FoundationStereo
  single ONNX (Google Drive release), needed only with `head_camera = "zed2"`
- `ffs/20_30_48_iters_8_res_576x960.engine` -- built from the ONNX on each deployment
  machine with `uv run python scripts/build_ffs_engine.py`

The artifact is git-ignored (~2 GB). The repository does not contain a placeholder
model file because a fake model would fail at runtime and could be mistaken for a
valid deployment artifact.
