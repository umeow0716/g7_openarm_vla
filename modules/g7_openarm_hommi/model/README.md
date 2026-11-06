Place the HoMMI ego3d deployment artifact here:

- `ego3d_policy.pt` -- produced by `uv run python scripts/convert_hommi_ckpt.py <latest.ckpt>`

The artifact is git-ignored (~2 GB). The repository does not contain a placeholder
model file because a fake model would fail at runtime and could be mistaken for a
valid deployment artifact.
