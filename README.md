# g7_openarm_vla

Joint/motor ordering contract: see [`JOINT_LAYOUT.md`](JOINT_LAYOUT.md).
Pre-deployment checks: see [`DEPLOYMENT_VALIDATION.md`](DEPLOYMENT_VALIDATION.md).

```bash
git clone --recurse-submodules https://github.com/umeow0716/g7_openarm_vla.git
cd g7_openarm_vla
```

```bash
uv sync --all-packages

# If unitree_sdk2py install failed, To setup cyclonedds as unitree_sdk2py README.md (https://github.com/unitreerobotics/unitree_sdk2_python)
# CYCLONEDDS_HOME=~/cyclonedds/install uv sync --all-packages

uv run scripts/run_sim.py
```


## HoMMI deployment

The integrated deployment module is `modules/g7_openarm_hommi`. Put the trained
artifacts under `modules/g7_openarm_hommi/model/` and use:

```bash
uv run python scripts/sim_hommi.py
uv run python scripts/real_hommi.py
```

The simulation launcher uses a 1920x1080 black RGB camera; the real launcher uses
a 1920x1080 `pyrealsense2` RGB stream. `[hommi].arm` must match the training
dataset's single-arm `arm_order`.

## Gripper coordinate convention

The gripper has exactly three coordinate spaces:

1. **Control / transport openness**: `q` is dimensionless in `[0.0, 1.0]`, where
   `0.0 = fully closed` and `1.0 = fully open`. `dq` is openness per second;
   positive velocity means opening.
2. **Model joint space**: each finger prismatic joint uses meters in
   `[0.0, 0.045]`. Model velocity is in m/s; the URDF velocity limit is
   `0.1 m/s`.
3. **Physical gripper motor space**: position is in radians and velocity is in
   rad/s. Open/close calibration endpoints are defined in `[hardware]` in
   `config.toml`.

Conversions between these spaces live in
`modules/g7_openarm_utils/src/g7_openarm_utils/gripper.py`.
