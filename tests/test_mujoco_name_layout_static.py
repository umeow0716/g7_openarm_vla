from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from g7_openarm_utils import (
    FLOATING_BASE_JOINT_NAME,
    MODEL_JOINTS_BY_MOTOR_NAME,
    MOTOR_NAMES,
)


def _model_root() -> ET.Element:
    path = Path(__file__).resolve().parents[1] / (
        "modules/g7_openarm_mujoco/src/g7_openarm_mujoco/model/g7_openarm.xml"
    )
    return ET.parse(path).getroot()


def test_mujoco_floating_base_has_stable_semantic_name() -> None:
    root = _model_root()
    freejoints = root.findall(".//worldbody//freejoint")
    assert len(freejoints) == 1
    assert freejoints[0].get("name") == FLOATING_BASE_JOINT_NAME


def test_every_logical_motor_joint_has_exactly_one_named_actuator() -> None:
    root = _model_root()
    model_joint_names = {
        joint.get("name") for joint in root.findall(".//worldbody//joint") if joint.get("name")
    }
    actuator_joint_names = [
        actuator.get("joint")
        for actuator in root.findall("./actuator/*")
        if actuator.get("joint")
    ]

    expected_joint_names = {
        joint_name
        for motor_name in MOTOR_NAMES
        for joint_name in MODEL_JOINTS_BY_MOTOR_NAME[motor_name]
    }
    assert expected_joint_names <= model_joint_names
    for joint_name in expected_joint_names:
        assert actuator_joint_names.count(joint_name) == 1


def test_every_logical_motor_has_named_primary_torque_sensor() -> None:
    root = _model_root()
    sensors = {
        sensor.get("name"): sensor
        for sensor in root.findall("./sensor/*")
        if sensor.get("name")
    }
    for motor_name in MOTOR_NAMES:
        sensor_name = f"{motor_name}_torque"
        assert sensor_name in sensors
        assert sensors[sensor_name].tag == "jointactuatorfrc"
        assert (
            sensors[sensor_name].get("joint")
            == MODEL_JOINTS_BY_MOTOR_NAME[motor_name][0]
        )


def test_urdf_and_mujoco_share_all_canonical_model_joint_names() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    urdf_path = repo_root / (
        "modules/g7_openarm_mujoco/src/g7_openarm_mujoco/model/urdf/g7_openarm.urdf"
    )
    urdf_root = ET.parse(urdf_path).getroot()
    urdf_joint_names = [
        joint.get("name") for joint in urdf_root.findall("./joint") if joint.get("name")
    ]
    assert len(urdf_joint_names) == len(set(urdf_joint_names))

    mujoco_root = _model_root()
    mujoco_joint_names = {
        joint.get("name")
        for joint in mujoco_root.findall(".//worldbody//joint")
        if joint.get("name")
    }
    expected_joint_names = {
        joint_name
        for motor_name in MOTOR_NAMES
        for joint_name in MODEL_JOINTS_BY_MOTOR_NAME[motor_name]
    }

    assert expected_joint_names <= set(urdf_joint_names)
    assert expected_joint_names <= mujoco_joint_names
