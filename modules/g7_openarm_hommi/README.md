# g7-openarm-hommi

Integrated HoMMI inference node for G7 OpenArm.

The node samples synchronized camera/robot observations at the configured HoMMI
frequency, keeps the model observation history, runs `hommi_train` inference in a
separate worker, converts the returned relative action chunk to absolute world
EEF targets using the FK frame captured by the newest observation, and publishes
those targets to `rt/eetarget` at a fixed 20 Hz.

## Model files

Place the deployment artifacts in `model/`:

```text
model/
├── model.pt
├── model.trt.eg
└── model.trt.eg.json
```

`model.trt.eg` is treated as the precompiled HoMMI TensorRT bundle. The current
`hommi_train` CLI normally names this bundle `model.trt.ep`; the loader is
extension-agnostic, so renaming the bundle to `model.trt.eg` is supported. If
`model.trt.eg` is absent, `model.trt.ep` is also accepted. If neither precompiled
bundle exists, the node falls back to `model.pt` and asks `hommi_train` to
configure the TensorRT backend at startup.

The JSON sidecar is retained beside the bundle for inspection; the runtime loader
uses the manifest embedded in the bundle itself.

## Camera

`RealSenseCamera` uses only the RGB stream at 1920x1080. `BlackCamera` implements
the same camera interface and returns an immutable 1920x1080 RGB black frame for
simulation.

`pyrealsense2` is imported only when `RealSenseCamera` is constructed, so the
simulation path does not require librealsense Python bindings.

## Control semantics

The current HoMMI artifact does not store the training dataset's `arm_order`.
Set `[hommi].arm` to the side used by the single-arm dataset.
For the included left-arm deployment config, use `general.control_mode = "left-arm"`;
`right-arm` is the corresponding setting for a right-arm model. `wbc` remains
accepted, but it also tracks the non-model arm target and is therefore not the
recommended single-arm mode.

For the canonical model used here:

- observation horizon: 2
- executable action chunk: 8
- action dimension: 10
- action row: `[relative_xyz(3), rotation6d(6), gripper_openness(1)]`

Every action row in one chunk is relative to the same newest EEF observation
frame. It must **not** be recomposed with fresh FK for every 20 Hz publish tick,
and action rows must **not** be chained together. The inference worker therefore
decodes the full chunk against the prediction-time FK frame before atomically
replacing the trajectory queue.

Use `scripts/sim_hommi.py` for the black-camera simulation path and
`scripts/real_hommi.py` for RealSense.
