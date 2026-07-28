from __future__ import annotations

import numpy as np

from g7_openarm_hardware.config import HardwareConfig
from g7_openarm_wbc.control_layout import control_size, split_control_vector


def _hardware_config() -> HardwareConfig:
    return HardwareConfig.from_mapping(
        {
            "dds": {"domain_id": 0, "interface": "lo"},
            "hardware": {
                "hz": 200.0,
                "imu_hz": 100,
                "base_can": "can_base",
                "left_arm_can": "can_left",
                "right_arm_can": "can_right",
                "can_fd": True,
                "base_ids": list(range(1, 9)),
                "base_direction": [1.0] * 8,
                "left_arm_direction": [1.0] * 8,
                "right_arm_direction": [1.0] * 8,
                "left_gripper_open": 0.0,
                "left_gripper_close": 1.0,
                "right_gripper_open": 0.0,
                "right_gripper_close": 1.0,
            },
        }
    )


def test_hardware_can_exclude_base_when_actuation_is_disabled() -> None:
    config = _hardware_config()
    assert config.active_can_interfaces(base_enabled=False) == ("can_left", "can_right")


def test_wbc_hardware_includes_base_can() -> None:
    config = _hardware_config()
    assert config.active_can_interfaces(base_enabled=True) == (
        "can_base",
        "can_left",
        "can_right",
    )


def test_arm_only_control_vector_contains_only_arm_dofs() -> None:
    assert control_size(base_enabled=False) == 14
    arm_u = np.arange(14, dtype=np.float64)

    amr_cmd, openarm_u = split_control_vector(arm_u, base_enabled=False)

    np.testing.assert_array_equal(amr_cmd, np.zeros(3))
    np.testing.assert_array_equal(openarm_u, arm_u)


def test_wbc_control_vector_splits_base_and_arms() -> None:
    assert control_size(base_enabled=True) == 17
    u = np.arange(17, dtype=np.float64)

    amr_cmd, openarm_u = split_control_vector(u, base_enabled=True)

    np.testing.assert_array_equal(amr_cmd, u[:3])
    np.testing.assert_array_equal(openarm_u, u[3:])


def test_base_only_hardware_uses_only_base_can() -> None:
    config = _hardware_config()
    assert config.active_can_interfaces(
        base_enabled=True,
        arms_enabled=False,
    ) == ("can_base",)
