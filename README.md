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

The integrated deployment module is `modules/g7_openarm_hommi` (HoMMI
`ego3d_lookat_policy`: bimanual + head look-at). Convert the training checkpoint
once, then launch:

```bash
uv run python scripts/convert_hommi_ckpt.py ~/Downloads/latest.ckpt
uv run python scripts/sim_hommi.py
uv run python scripts/real_hommi.py
```

The real launcher needs the D435i serials in `[hommi]` of `config.toml`.

### ZED 2 head camera

With `head_camera = "zed2"` the head is a ZED 2 (stereo + Fast-FoundationStereo) and only
the two wrist D435i serials are needed, plus:

- `zed_device`: V4L2 index or device path of the ZED 2.
- `zed_calibration`: the factory calibration `.conf` for your camera serial, from
  [calib.stereolabs.com](https://calib.stereolabs.com/).

Download `onnx/20_30_48/576x960/20_30_48_iters_8_res_576x960.onnx` and its `.yaml` from the
official [Fast-FoundationStereo Google Drive](https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link)
into `modules/g7_openarm_hommi/model/ffs/`, then build the TensorRT engine once on each
machine (engines only run on the GPU and TensorRT version that built them):

```bash
uv run python scripts/build_ffs_engine.py
```

See `modules/g7_openarm_hommi/README.md` for camera, frame and action details.

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
