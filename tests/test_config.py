from __future__ import annotations

import pytest

from g7_openarm_config import DDSConfig, GeneralConfig, parse_bool
from g7_openarm_hardware.config import HardwareConfig
from g7_openarm_lowlevel.config import ControlMode, LowLevelConfig


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("YES", True),
        ("0", False),
        (" off ", False),
    ],
)
def test_parse_bool_accepts_only_explicit_values(value: object, expected: bool) -> None:
    assert parse_bool(value, field="test.value") is expected


@pytest.mark.parametrize("value", [1, 0, [], "enabled", None])
def test_parse_bool_rejects_ambiguous_values(value: object) -> None:
    with pytest.raises(ValueError, match="test.value"):
        parse_bool(value, field="test.value")


def test_general_config_does_not_treat_false_string_as_true() -> None:
    config = GeneralConfig.from_mapping({"general": {"debugging": "false"}})
    assert config.debugging is False


def test_dds_config_trims_interface_and_validates_domain() -> None:
    config = DDSConfig.from_mapping({"dds": {"domain_id": 3, "interface": " enp2s0 "}})
    assert config.domain_id == 3
    assert config.interface == "enp2s0"

    with pytest.raises(ValueError, match="domain_id"):
        DDSConfig.from_mapping({"dds": {"domain_id": -1, "interface": "lo"}})


def test_control_mode_is_strict_but_normalized() -> None:
    data = {
        "dds": {"domain_id": 0, "interface": "lo"},
        "lowlevel": {"hz": 200.0, "control_mode": " WBC "},
    }
    config = LowLevelConfig.from_mapping(data)
    assert config.control_mode is ControlMode.WBC
    assert config.base_enabled is True

    data["lowlevel"]["control_mode"] = "arm-only"
    config = LowLevelConfig.from_mapping(data)
    assert config.control_mode is ControlMode.ARM_ONLY
    assert config.base_enabled is False

    data["lowlevel"]["control_mode"] = "wbc-typo"
    with pytest.raises(ValueError, match="control_mode"):
        LowLevelConfig.from_mapping(data)


def test_hardware_config_rejects_invalid_motor_mapping() -> None:
    data = {
        "dds": {"domain_id": 0, "interface": "lo"},
        "hardware": {
            "hz": 200.0,
            "imu_hz": 100,
            "base_can": "can_base",
            "left_arm_can": "can_left",
            "right_arm_can": "can_right",
            "can_fd": "true",
            "base_ids": [1, 2, 3, 4, 5, 6, 7, 7],
            "base_direction": [1.0] * 8,
            "left_arm_direction": [1.0] * 8,
            "right_arm_direction": [1.0] * 8,
        },
    }

    with pytest.raises(ValueError, match="permutation"):
        HardwareConfig.from_mapping(data)
