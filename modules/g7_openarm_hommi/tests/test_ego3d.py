import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g7_openarm_hommi.camera import PinholeIntrinsics
from g7_openarm_hommi.crop import pointmap_mask
from g7_openarm_hommi.geometry import matrix_to_rotation_6d_umi
from g7_openarm_hommi.gripper import GripperWidthRange
from g7_openarm_hommi.state import COLOR_LINK_TO_OPTICAL, RobotSnapshot
from g7_openarm_hommi.trajectory import ARM_SLICES, LOOKAT, decode_action_chunk
from g7_openarm_hommi.vcam import (
    IPHONE_DEPTH,
    IPHONE_WIDE,
    depth_to_xyz_map,
    resample_rgb_to_virtual_square,
    splat_depth_to_virtual,
)

D435_COLOR = PinholeIntrinsics(1920, 1080, 1380.0, 1378.0, 955.0, 545.0)


def pose(rotvec, translation):
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    out[:3, 3] = translation
    return out


def reference_resample(image, source, output_size, target=IPHONE_WIDE):
    """Slow reference: remap into the virtual camera, pad to square, INTER_AREA."""
    us, vs = np.meshgrid(np.arange(target.width), np.arange(target.height))
    map_x = ((us - target.cx) / target.fx * source.fx + source.cx).astype(np.float32)
    map_y = ((vs - target.cy) / target.fy * source.fy + source.cy).astype(np.float32)
    virtual = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    side = max(target.width, target.height)
    top = (side - target.height) // 2
    left = (side - target.width) // 2
    padded = cv2.copyMakeBorder(
        virtual,
        top,
        side - target.height - top,
        left,
        side - target.width - left,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    return cv2.resize(padded, (output_size, output_size), interpolation=cv2.INTER_AREA)


@pytest.mark.parametrize("output_size", [224, 512])
def test_rgb_resampling_matches_remap_reference(output_size):
    rng = np.random.default_rng(0)
    # Smooth image so sub-pixel offsets do not dominate the comparison.
    small = rng.integers(0, 255, (27, 48, 3), dtype=np.uint8)
    image = cv2.resize(small, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    fast = resample_rgb_to_virtual_square(image, D435_COLOR, output_size=output_size)
    ref = reference_resample(image, D435_COLOR, output_size)
    assert fast.shape == ref.shape == (output_size, output_size, 3)

    def extent(img, axis):
        lit = np.nonzero(img.max(axis=(axis, 2)) > 0)[0]
        return lit[0], lit[-1]

    # Valid-region borders agree to within one output pixel (partial-coverage edge).
    for axis in (0, 1):
        np.testing.assert_allclose(extent(fast, axis), extent(ref, axis), atol=1)
    # Interior content agrees up to the <= 0.5 px sub-pixel shift of the fast path.
    r0, r1 = extent(ref, 1)
    c0, c1 = extent(ref, 0)
    inner = (slice(r0 + 2, r1 - 1), slice(c0 + 2, c1 - 1))
    assert np.mean(np.abs(fast[inner].astype(int) - ref[inner].astype(int))) < 3.0


def test_rgb_resampling_keeps_training_padding_black():
    image = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    out = resample_rgb_to_virtual_square(image, D435_COLOR, output_size=224)
    pad = round(240 * 224 / 1920)  # iPhone 1920x1440 -> 1920x1920 square
    assert out[: pad - 1].max() == 0
    assert out[-(pad - 1) :].max() == 0
    # D435i's narrower vertical FOV leaves extra black rows inside the iPhone frame.
    assert out[pad + 2].max() == 0
    # ... and a few columns (D435i HFOV ~69.6 deg < iPhone 72.7 deg).
    assert out[112, 0].max() == 0
    assert out[112, 10:-10].min() == 255


def test_depth_splat_of_fronto_parallel_plane():
    depth_intr = PinholeIntrinsics(848, 480, 425.0, 425.0, 424.0, 240.0)
    depth = np.full((480, 848), 1.5, dtype=np.float32)
    out = splat_depth_to_virtual(depth, depth_intr, np.eye(4))
    assert out.shape == (IPHONE_DEPTH.height, IPHONE_DEPTH.width)
    covered = out > 0
    assert covered.mean() > 0.8
    np.testing.assert_allclose(out[covered], 1.5, atol=1e-5)
    xyz = depth_to_xyz_map(out, IPHONE_DEPTH)
    np.testing.assert_allclose(
        xyz[96, 127], [-0.1 * 1.5 / 173.8, -0.5 * 1.5 / 173.8, 1.5], atol=1e-4
    )


def test_depth_splat_applies_extrinsics_and_zbuffer():
    depth_intr = PinholeIntrinsics(848, 480, 425.0, 425.0, 424.0, 240.0)
    depth = np.full((480, 848), 2.0, dtype=np.float32)
    shift = np.eye(4)
    shift[2, 3] = -0.5  # color frame is 0.5 m in front of the depth sensor
    out = splat_depth_to_virtual(depth, depth_intr, shift)
    np.testing.assert_allclose(out[out > 0], 1.5, atol=1e-5)


def test_pointmap_mask_depth_range_and_gripper_crop():
    head = np.eye(4, dtype=np.float32)
    pts = np.zeros((1, 4, 3), dtype=np.float32)
    pts[0, :, 2] = [0.05, 1.0, 5.0, 0.3]
    # Gripper TCP at z=0.5 looking along +z: point 3 (z=0.3) is 0.2 m behind it.
    tcp = pose([0, 0, 0], [0.0, 0.0, 0.5])
    far_tcp = pose([0, 0, 0], [0.0, 0.0, -10.0])
    mask = pointmap_mask(pts, head, (tcp, far_tcp), depth_range=(0.1, 4.0), gripper_length=0.13)
    assert mask.tolist() == [[False, True, False, False]]


def test_head_optical_frame_axes():
    r = COLOR_LINK_TO_OPTICAL[:3, :3]
    np.testing.assert_allclose(r[:, 2], [1, 0, 0])  # optical z = link x (forward)
    np.testing.assert_allclose(r[:, 0], [0, -1, 0])  # optical x = -link y (right)
    np.testing.assert_allclose(r[:, 1], [0, 0, -1])  # optical y = -link z (down)


def test_gripper_width_range_roundtrip():
    rng = GripperWidthRange(closed=0.0, opened=0.25)
    assert rng.to_width(0.0) == 0.0
    assert rng.to_width(1.0) == pytest.approx(0.25)
    assert rng.to_openness(0.125) == pytest.approx(0.5)
    assert rng.to_openness(0.4) == 1.0
    assert rng.to_openness(-0.1) == 0.0
    with pytest.raises(ValueError):
        GripperWidthRange(closed=0.2, opened=0.1)


def test_decode_recovers_world_targets():
    left = pose([0.1, -0.2, 0.3], [0.4, 0.2, 0.9])
    right = pose([-0.3, 0.1, 0.2], [0.4, -0.2, 0.9])
    snapshot = RobotSnapshot(
        timestamp=10.0,
        left_tcp=left,
        right_tcp=right,
        head=np.eye(4, dtype=np.float32),
        left_gripper_width=0.0,
        right_gripper_width=0.0,
    )
    goals = {
        "left": pose([0.2, 0.0, 0.1], [0.5, 0.1, 0.8]),
        "right": pose([0, 0.3, 0], [0.5, -0.1, 0.7]),
    }
    look_at = np.array([0.8, 0.0, 0.6])

    action = np.zeros((1, 23), dtype=np.float32)
    base_inv = np.linalg.inv(left.astype(np.float64))
    for side, (pos, rot, width) in ARM_SLICES.items():
        rel = base_inv @ goals[side]
        action[0, pos] = rel[:3, 3]
        action[0, rot] = matrix_to_rotation_6d_umi(rel[:3, :3])
        action[0, width] = 0.1
    action[0, LOOKAT] = base_inv[:3, :3] @ look_at + base_inv[:3, 3]

    ranges = {"left": GripperWidthRange(0.0, 0.2), "right": GripperWidthRange(0.0, 0.4)}
    (point,) = decode_action_chunk(
        action, reference=snapshot, gripper_ranges=ranges, interval=0.05, obs_horizon=2
    )
    for side, pose7 in (("left", point.left_pose7), ("right", point.right_pose7)):
        np.testing.assert_allclose(pose7[:3], goals[side][:3, 3], atol=1e-5)
        quat_wxyz = np.asarray(pose7[3:])
        got = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(got, goals[side][:3, :3], atol=1e-5)
    np.testing.assert_allclose(point.look_at_point, look_at, atol=1e-5)
    assert point.left_gripper_openness == pytest.approx(0.5)
    assert point.right_gripper_openness == pytest.approx(0.25)
    assert point.execute_at == pytest.approx(10.05)


def _chunk(t0, n=8, dt=0.05, x0=0.0):
    from g7_openarm_hommi.trajectory import TargetPoint

    ident = (1.0, 0.0, 0.0, 0.0)
    return [
        TargetPoint(
            t0 + i * dt,
            (x0 + i, 0.0, 0.0, *ident),
            (0.0, 0.0, 0.0, *ident),
            i / 10,
            0.0,
            (0.0, 0.0, 0.0),
        )
        for i in range(n)
    ]


def test_sample_interpolates_between_steps():
    from g7_openarm_hommi.trajectory import AtomicTrajectory

    traj = AtomicTrajectory()
    assert traj.sample(now=10.0) is None
    traj.replace(_chunk(10.05), now=10.0, latency_compensation=True)
    mid = traj.sample(now=10.075)  # halfway between steps 0 and 1
    assert mid.left_pose7[0] == pytest.approx(0.5)
    assert mid.left_gripper_openness == pytest.approx(0.05)
    assert traj.sample(now=99.0).left_pose7[0] == pytest.approx(7.0)  # holds the last step


def test_slerp_is_shortest_arc_and_hemisphere_safe():
    from g7_openarm_hommi.trajectory import _slerp_wxyz

    q0 = Rotation.from_euler("z", 170, degrees=True).as_quat()[[3, 0, 1, 2]]
    q1 = -Rotation.from_euler("z", -170, degrees=True).as_quat()[[3, 0, 1, 2]]  # flipped sign
    mid = _slerp_wxyz(q0, q1, 0.5)
    yaw = Rotation.from_quat(mid[[1, 2, 3, 0]]).as_euler("xyz", degrees=True)[2]
    assert abs(abs(yaw) - 180.0) < 1e-6  # 20 deg arc through 180, not 340 deg through 0


def test_chunk_switch_starts_from_current_target():
    from g7_openarm_hommi.trajectory import AtomicTrajectory

    traj = AtomicTrajectory()
    traj.replace(_chunk(10.0), now=10.0, latency_compensation=True)
    current = traj.sample(now=10.12).left_pose7[0]  # 2.4 on the old chunk
    expired = traj.replace(_chunk(10.0, x0=100.0), now=10.12, latency_compensation=True)
    assert expired == 3
    # No jump at the switch; then it blends into the new chunk's first future step.
    assert traj.sample(now=10.12).left_pose7[0] == pytest.approx(current)
    assert traj.sample(now=10.15).left_pose7[0] == pytest.approx(103.0)


def test_fully_late_chunk_blends_to_final_step():
    from g7_openarm_hommi.trajectory import AtomicTrajectory

    traj = AtomicTrajectory()
    traj.replace(_chunk(10.0), now=10.0, latency_compensation=True)
    held = traj.sample(now=12.0).left_pose7[0]
    assert traj.replace(_chunk(11.0, x0=100.0), now=12.0, latency_compensation=True) == 8
    assert traj.sample(now=12.0).left_pose7[0] == pytest.approx(held)
    assert traj.sample(now=12.025).left_pose7[0] == pytest.approx((held + 107.0) / 2)
    assert traj.sample(now=12.05).left_pose7[0] == pytest.approx(107.0)


def test_without_latency_compensation_chunk_restarts_now():
    from g7_openarm_hommi.trajectory import AtomicTrajectory

    traj = AtomicTrajectory()
    assert traj.replace(_chunk(10.0), now=12.0, latency_compensation=False) == 0
    assert traj.sample(now=12.05).left_pose7[0] == pytest.approx(0.0)
    assert traj.sample(now=12.10).left_pose7[0] == pytest.approx(1.0)


def test_point_cloud_crop_follows_checkpoint_settings():
    from g7_openarm_hommi.crop import PointCloudCrop

    head = np.eye(4, dtype=np.float32)
    pts = np.zeros((1, 3, 3), dtype=np.float32)
    pts[0, :, 2] = [0.05, 1.0, 0.3]
    tcp = pose([0, 0, 0], [0.0, 0.0, 0.5])  # point 2 is 0.2 m behind this TCP
    crop = PointCloudCrop(enabled=True, depth_range=(0.1, 4.0), gripper_length=0.13)
    assert crop.mask(pts, head, (tcp,)).tolist() == [[False, True, False]]
    longer = PointCloudCrop(enabled=True, depth_range=(0.1, 4.0), gripper_length=0.25)
    assert longer.mask(pts, head, (tcp,)).tolist() == [[False, True, True]]
    assert PointCloudCrop(enabled=False).mask(pts, head, (tcp,)).all()


def test_runner_reads_crop_from_checkpoint_config():
    """The ego3d checkpoint enables cropping with depth 0.1-4.0 m and the 0.13 m default."""
    import torch

    from g7_openarm_hommi.policy import DEFAULT_ARTIFACT_NAME, module_model_dir

    path = module_model_dir() / DEFAULT_ARTIFACT_NAME
    if not path.is_file():
        pytest.skip("ego3d artifact not converted")
    cfg = torch.load(path, map_location="cpu", weights_only=True)["cfg"]["task"]["dataset"]
    assert cfg["crop_pointcloud"] is True
    assert list(cfg["depth_range"]) == [0.1, 4.0]
    assert "gripper_length" not in cfg  # -> HoMMI default 0.13
