# Deployment validation checklist

The refactor makes semantic motor/joint names the public contract and confines
integer indices to validated protocol/model boundaries. The checks below are
still required before commanding real hardware because the review container did
not contain the project's full Python 3.12 + MuJoCo + CycloneDDS + Pinocchio
runtime.

## 1. Regenerate PinnZoo v2 libraries

The supplied G7 `.so` files remain compatible legacy API v1 binaries. They
already export q/v/torque name arrays and the refactored binding consumes those
arrays by name, but API v2 adds joint metadata and direct lookup functions.

On the machine that has Python Pinocchio and CasADi, use the updated PinnZoo
source package:

```bash
python models/g7_openarm/sync_urdf.py \
  <g7-repo>/modules/g7_openarm_mujoco/src/g7_openarm_mujoco/model/urdf/g7_openarm.urdf
python models/g7_openarm/generate.py
cmake -S . -B build
cmake --build build --target g7_openarm_quat
```

Build and copy the library for each deployment architecture separately. The
G7 binding accepts v1 and v2, and performs stronger metadata cross-checks for
v2.

## 2. Run the project in its declared Python 3.12 environment

```bash
uv sync --all-packages
uv run pytest -q
```

Do not treat a partial test run as equivalent to this full environment test.
The source review environment used Python 3.13 and could not execute tests that
import MuJoCo or CycloneDDS.

## 3. MuJoCo smoke test

Start the normal simulator and verify startup does not raise a layout error:

```bash
uv run scripts/run_sim.py
```

`MuJoCoModelLayout` resolves every required joint, independent qpos/qvel
address, actuator, and torque sensor by name at startup. A missing or ambiguous
mapping is intentionally fatal instead of silently using a guessed offset.

## 4. Hardware dry-run before enabling motion

Before real motion, verify the configured CAN interfaces, `base_ids`, direction
arrays, arm buses, and emergency-stop path. First inspect `rt/lowstate` with
actuation disabled and confirm named motors report the expected physical axes.
Only then enable commands with the normal project safety procedure.

## 5. Protocol rule

Do not add raw numeric motor/q/v slices to first-party control code. If a future
process is versioned independently from this workspace and therefore cannot
share `g7_openarm_utils.joint_layout`, add a startup-only CycloneDDS layout
handshake (name + layout version/hash). Do not send joint-name strings every
realtime control tick; validate once and cache indices.
