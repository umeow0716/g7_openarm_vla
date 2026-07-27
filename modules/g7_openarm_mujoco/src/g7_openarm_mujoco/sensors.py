from __future__ import annotations

import mujoco


def sensor_slice(model: mujoco.MjModel, name: str, *, expected_dim: int | None = None) -> slice:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise KeyError(f"MuJoCo sensor not found: {name}")

    address = int(model.sensor_adr[sensor_id])
    dimension = int(model.sensor_dim[sensor_id])

    if expected_dim is not None and dimension != expected_dim:
        raise ValueError(
            f"MuJoCo sensor {name!r} has dimension {dimension}, expected {expected_dim}"
        )

    return slice(address, address + dimension)


def scalar_sensor_address(model: mujoco.MjModel, name: str) -> int:
    return sensor_slice(model, name, expected_dim=1).start
