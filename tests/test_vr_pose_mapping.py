import numpy as np

from g7_openarm_utils import (
    quat_from_yaw,
    quat_mul,
    quat_to_rotation_matrix,
)
from g7_openarm_vr.pose_mapping import RelativePoseMapper
from g7_openarm_vr.udp_response import (
    VR_RH_TO_MUJOCO_QUAT,
    VRControllerPose,
    VRUDPResponse,
)


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
    np.testing.assert_allclose(_unity_pose(z=1.0).as_mujoco_pose()[:3], [1, 0, 0])
    np.testing.assert_allclose(_unity_pose(x=1.0).as_mujoco_pose()[:3], [0, -1, 0])
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


def test_mapper_maps_captured_pose_exactly_to_robot_origin() -> None:
    first = np.concatenate(([1.0, 2.0, 3.0], quat_from_yaw(0.4)))
    origin = np.concatenate(([10.0, 20.0, 30.0], quat_from_yaw(-0.7)))

    mapper = RelativePoseMapper.from_poses(first, origin)

    np.testing.assert_allclose(mapper.map(first), origin, atol=1e-12)


def test_mapper_preserves_world_axis_translation_directions() -> None:
    first = np.concatenate(([1.0, 2.0, 3.0], quat_from_yaw(0.8)))
    origin = np.concatenate(([10.0, 20.0, 30.0], quat_from_yaw(-0.3)))
    mapper = RelativePoseMapper.from_poses(first, origin)

    forward = first.copy()
    forward[0] += 0.2
    right = first.copy()
    right[1] -= 0.2
    up = first.copy()
    up[2] += 0.2

    np.testing.assert_allclose(mapper.map(forward)[:3], origin[:3] + [0.2, 0, 0])
    np.testing.assert_allclose(mapper.map(right)[:3], origin[:3] + [0, -0.2, 0])
    np.testing.assert_allclose(mapper.map(up)[:3], origin[:3] + [0, 0, 0.2])


def test_mapper_applies_rotation_relative_to_captured_orientation() -> None:
    first_q = quat_from_yaw(0.4)
    origin_q = quat_from_yaw(-0.7)
    relative_rotation = quat_from_yaw(0.25)
    current_q = quat_mul(relative_rotation, first_q)

    first = np.concatenate(([1.0, 2.0, 3.0], first_q))
    origin = np.concatenate(([10.0, 20.0, 30.0], origin_q))
    current = np.concatenate(([1.0, 2.0, 3.0], current_q))

    target = RelativePoseMapper.from_poses(first, origin).map(current)
    expected_q = quat_mul(relative_rotation, origin_q)

    np.testing.assert_allclose(
        quat_to_rotation_matrix(target[3:]),
        quat_to_rotation_matrix(expected_q),
        atol=1e-12,
    )
