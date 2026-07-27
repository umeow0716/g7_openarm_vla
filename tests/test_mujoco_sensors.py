from __future__ import annotations

import pytest

from g7_openarm_mujoco.resources import model_directory
from g7_openarm_mujoco.sensors import scalar_sensor_address, sensor_slice

mujoco = pytest.importorskip("mujoco")


def test_sensor_helpers_use_sensor_addresses_not_sensor_ids() -> None:
    with model_directory() as model_dir:
        model = mujoco.MjModel.from_xml_path((model_dir / "scene.xml").as_posix())

    quaternion = sensor_slice(model, "imu_quat", expected_dim=4)
    gyroscope = sensor_slice(model, "imu_gyro", expected_dim=3)
    accelerometer = sensor_slice(model, "imu_acc", expected_dim=3)

    assert quaternion.stop - quaternion.start == 4
    assert gyroscope.stop - gyroscope.start == 3
    assert accelerometer.stop - accelerometer.start == 3
    assert quaternion.stop == gyroscope.start
    assert gyroscope.stop == accelerometer.start

    motor_address = scalar_sensor_address(model, "AMR_FL_pos")
    motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "AMR_FL_pos")
    assert motor_address == int(model.sensor_adr[motor_id])
