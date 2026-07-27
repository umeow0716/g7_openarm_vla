import numpy as np
import pytest

from g7_openarm_utils import quat_from_yaw, quat_to_rotation_matrix, quat_yaw
from g7_openarm_vr.udp_response import (
    VR_RH_TO_MUJOCO_QUAT,
    VRControllerPose,
    VRUDPResponse,
)
from g7_openarm_vr.pose_mapping import remap_pose_yaw_only


def _unity_pose(*, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> VRControllerPose:
    return VRControllerPose(x=x, y=y, z=z, qw=1.0, qx=0.0, qy=0.0, qz=0.0)


def test_udp_response_parses_controller_objects() -> None:
    packet = {
        "lc": {"x": 1, "y": 2, "z": 3, "qw": 1, "qx": 0, "qy": 0, "qz": 0},
        "rc": {"x": 4, "y": 5, "z": 6, "qw": 1, "qx": 0, "qy": 0, "qz": 0},
    }

    response = VRUDPResponse.from_mapping(packet)

    assert isinstance(response.left_controller, VRControllerPose)
    np.testing.assert_allclose(
        response.left_controller.as_mujoco_pose()[:3],
        [3.0, -1.0, 2.0],
    )


def test_vr_position_axes_map_to_mujoco_forward_left_up() -> None:
    # Unity +Z forward -> MuJoCo +X forward.
    np.testing.assert_allclose(_unity_pose(z=1.0).as_mujoco_pose()[:3], [1, 0, 0])
    # Unity +X right -> MuJoCo -Y (right).
    np.testing.assert_allclose(_unity_pose(x=1.0).as_mujoco_pose()[:3], [0, -1, 0])
    # Unity +Y up -> MuJoCo +Z up.
    np.testing.assert_allclose(_unity_pose(y=1.0).as_mujoco_pose()[:3], [0, 0, 1])


def test_identity_unity_orientation_gets_fixed_mujoco_basis_rotation() -> None:
    pose = _unity_pose().as_mujoco_pose()
    expected = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(
        quat_to_rotation_matrix(pose[3:]),
        expected,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        quat_to_rotation_matrix(VR_RH_TO_MUJOCO_QUAT),
        expected,
        atol=1e-12,
    )


def test_yaw_only_mapping_rotates_translation_and_preserves_relative_pitch_roll() -> None:
    first = np.concatenate(([1.0, 2.0, 3.0], quat_from_yaw(0.0)))
    origin = np.concatenate(([10.0, 20.0, 30.0], quat_from_yaw(np.pi / 2.0)))
    current = np.concatenate(([2.0, 2.0, 3.0], quat_from_yaw(0.25)))

    target = remap_pose_yaw_only(current, first, origin)

    np.testing.assert_allclose(target[:3], [10.0, 21.0, 30.0], atol=1e-12)
    assert quat_yaw(target[3:]) == pytest.approx(np.pi / 2.0 + 0.25)
