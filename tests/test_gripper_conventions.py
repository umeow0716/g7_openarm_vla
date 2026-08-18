from __future__ import annotations

import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from g7_openarm_utils import gripper as gripper_module
from g7_openarm_utils.gripper import (
    GRIPPER_MODEL_OPEN_DISTANCE_M,
    GRIPPER_MODEL_VELOCITY_LIMIT_M_S,
    GRIPPER_OPENNESS_VELOCITY_LIMIT_PER_S,
    gripper_model_position_to_openness,
    gripper_model_velocity_to_openness_velocity,
    gripper_motor_position_to_openness,
    gripper_motor_velocity_to_openness_velocity,
    gripper_openness_to_model_position,
    gripper_openness_to_motor_position,
    gripper_openness_velocity_to_model_velocity,
    gripper_openness_velocity_to_motor_velocity,
)

ROOT = Path(__file__).resolve().parents[1]


def _hardware_gripper_calibrations() -> dict[str, tuple[float, float]]:
    with (ROOT / "config.toml").open("rb") as stream:
        hardware = tomllib.load(stream)["hardware"]
    return {
        "left": (
            float(hardware["left_gripper_open"]),
            float(hardware["left_gripper_close"]),
        ),
        "right": (
            float(hardware["right_gripper_open"]),
            float(hardware["right_gripper_close"]),
        ),
    }


def test_only_three_gripper_coordinate_spaces_remain() -> None:
    """There is no legacy gripper transport coordinate anywhere in source."""
    source = "\n".join(path.read_text() for path in (ROOT / "modules").rglob("*.py"))
    assert "GRIPPER_COMMAND_RANGE" not in source
    assert "gripper_command" not in source
    assert ("0." + "45") not in source
    assert not hasattr(gripper_module, "GRIPPER_COMMAND_RANGE")


def test_openness_endpoints_match_model_and_hardware_spaces() -> None:
    """Canonical transport API is openness: 0=closed, 1=open."""
    assert gripper_openness_to_model_position(0.0) == pytest.approx(0.0)
    assert gripper_openness_to_model_position(1.0) == pytest.approx(
        GRIPPER_MODEL_OPEN_DISTANCE_M
    )

    for open_position, close_position in _hardware_gripper_calibrations().values():
        assert gripper_openness_to_motor_position(
            0.0,
            open_position=open_position,
            close_position=close_position,
        ) == pytest.approx(close_position)
        assert gripper_openness_to_motor_position(
            1.0,
            open_position=open_position,
            close_position=close_position,
        ) == pytest.approx(open_position)


@pytest.mark.parametrize("openness", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_model_and_actual_hardware_positions_round_trip_through_openness(
    openness: float,
) -> None:
    model_position_m = gripper_openness_to_model_position(openness)
    assert gripper_model_position_to_openness(model_position_m) == pytest.approx(openness)

    for open_position, close_position in _hardware_gripper_calibrations().values():
        motor_position_rad = gripper_openness_to_motor_position(
            openness,
            open_position=open_position,
            close_position=close_position,
        )
        restored_openness = gripper_motor_position_to_openness(
            motor_position_rad,
            open_position=open_position,
            close_position=close_position,
        )
        assert restored_openness == pytest.approx(openness)


@pytest.mark.parametrize("openness_velocity", [-1.0, -0.25, 0.25, 1.0])
def test_velocity_mapping_is_derivative_of_position_mapping(
    openness_velocity: float,
) -> None:
    """dq>0 always means opening in the normalized command space."""
    model_velocity_m_s = gripper_openness_velocity_to_model_velocity(openness_velocity)
    assert model_velocity_m_s == pytest.approx(
        GRIPPER_MODEL_OPEN_DISTANCE_M * openness_velocity
    )
    assert gripper_model_velocity_to_openness_velocity(model_velocity_m_s) == pytest.approx(
        openness_velocity
    )

    for open_position, close_position in _hardware_gripper_calibrations().values():
        motor_velocity_rad_s = gripper_openness_velocity_to_motor_velocity(
            openness_velocity,
            open_position=open_position,
            close_position=close_position,
        )
        assert motor_velocity_rad_s == pytest.approx(
            (open_position - close_position) * openness_velocity
        )
        restored_openness_velocity = gripper_motor_velocity_to_openness_velocity(
            motor_velocity_rad_s,
            open_position=open_position,
            close_position=close_position,
        )
        assert restored_openness_velocity == pytest.approx(openness_velocity)


def test_normalized_velocity_limit_matches_model_velocity_limit() -> None:
    assert (
        gripper_openness_velocity_to_model_velocity(
            GRIPPER_OPENNESS_VELOCITY_LIMIT_PER_S
        )
        == pytest.approx(GRIPPER_MODEL_VELOCITY_LIMIT_M_S)
    )


def test_mjcf_and_urdf_gripper_joint_limits_match_model_convention() -> None:
    expected_joint_names = {
        "gripper_LL_joint",
        "gripper_LR_joint",
        "gripper_RL_joint",
        "gripper_RR_joint",
    }

    mjcf = ET.parse(
        ROOT / "modules/g7_openarm_mujoco/src/g7_openarm_mujoco/model/g7_openarm.xml"
    ).getroot()
    mjcf_joints = {
        joint.attrib["name"]: joint
        for joint in mjcf.iter("joint")
        if joint.attrib.get("name") in expected_joint_names
    }
    assert set(mjcf_joints) == expected_joint_names
    for joint in mjcf_joints.values():
        lower, upper = map(float, joint.attrib["range"].split())
        assert lower == pytest.approx(0.0)
        assert upper == pytest.approx(GRIPPER_MODEL_OPEN_DISTANCE_M)

    urdf = ET.parse(
        ROOT / "modules/g7_openarm_mujoco/src/g7_openarm_mujoco/model/urdf/g7_openarm.urdf"
    ).getroot()
    urdf_joints = {
        joint.attrib["name"]: joint
        for joint in urdf.findall("joint")
        if joint.attrib.get("name") in expected_joint_names
    }
    assert set(urdf_joints) == expected_joint_names
    for joint in urdf_joints.values():
        limit = joint.find("limit")
        assert limit is not None
        assert float(limit.attrib["lower"]) == pytest.approx(0.0)
        assert float(limit.attrib["upper"]) == pytest.approx(GRIPPER_MODEL_OPEN_DISTANCE_M)
        assert float(limit.attrib["velocity"]) == pytest.approx(
            GRIPPER_MODEL_VELOCITY_LIMIT_M_S
        )
