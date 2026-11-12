from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g7_openarm_hommi.observation import ObservationSample
from g7_openarm_hommi.state import (
    COLOR_LINK_TO_OPTICAL,
    RobotSnapshot,
    StateHistory,
    head_camera_extrinsic,
)
from g7_openarm_hommi.vcam import HeadObservation


def test_without_rectification_is_the_d435i_color_optical_frame():
    np.testing.assert_allclose(head_camera_extrinsic(), COLOR_LINK_TO_OPTICAL, atol=1e-7)


def test_rectification_maps_rectified_points_back_to_raw_camera():
    r1 = Rotation.from_euler("xyz", [0.01, -0.02, 0.005]).as_matrix()
    ext = head_camera_extrinsic(r1).astype(np.float64)
    p_raw = np.array([0.1, -0.05, 1.0])
    p_rect = r1 @ p_raw
    via_rect = ext @ np.append(p_rect, 1.0)
    via_raw = COLOR_LINK_TO_OPTICAL.astype(np.float64) @ np.append(p_raw, 1.0)
    np.testing.assert_allclose(via_rect, via_raw, atol=1e-6)


def test_state_history_returns_nearest_messages():
    history = StateHistory(horizon_s=1.0)
    assert history.latest() is None
    for i in range(10):
        history.add_lowstate(10.0 + 0.002 * i, f"low{i}")
    history.add_odom(10.000, "odom0")
    history.add_odom(10.010, "odom1")
    assert history.latest() == ("low9", "odom1")
    assert history.at(10.0071) == ("low4", "odom1")
    assert history.at(10.0049) == ("low2", "odom0")
    assert history.at(0.0) == ("low0", "odom0")


def test_state_history_drops_messages_older_than_horizon():
    history = StateHistory(horizon_s=0.5)
    history.add_odom(0.0, "odom")
    history.add_lowstate(0.0, "old")
    history.add_lowstate(1.0, "new")
    history.add_odom(1.0, "odom_new")
    assert history.at(0.0) == ("new", "odom_new")


def _snapshot(head_x: float) -> RobotSnapshot:
    head = np.eye(4, dtype=np.float32)
    head[0, 3] = head_x
    eye = np.eye(4, dtype=np.float32)
    return RobotSnapshot(0.0, eye, eye, head, 0.05, 0.05)


@pytest.mark.parametrize("with_head_robot", [False, True])
def test_head_state_prefers_capture_time_snapshot(with_head_robot):
    head = HeadObservation(
        rgb=np.zeros((4, 4, 3), np.uint8), pointmap=np.zeros((4, 4, 3), np.float32)
    )
    now, captured = _snapshot(1.0), _snapshot(2.0)
    sample = ObservationSample(
        sequence=1,
        timestamp=0.0,
        head=head,
        wrist_rgb={},
        robot=now,
        head_robot=captured if with_head_robot else None,
    )
    assert sample.head_state is (captured if with_head_robot else now)


@pytest.mark.parametrize(
    "intr",
    [
        (640, 360, 282.8, 282.8, 319.8, 179.7),
        (848, 480, 425.0, 425.0, 424.0, 240.0),
    ],
)
def test_depth_splat_has_no_holes_inside_the_covered_region(intr):
    from g7_openarm_hommi.camera import PinholeIntrinsics
    from g7_openarm_hommi.vcam import splat_depth_to_virtual

    source = PinholeIntrinsics(*intr)
    depth = np.ones((source.height, source.width), np.float32)
    out = splat_depth_to_virtual(depth, source, np.eye(4)) > 0
    rows, cols = np.where(out.any(axis=1))[0], np.where(out.any(axis=0))[0]
    inner = out[rows[0] + 1 : rows[-1], cols[0] + 1 : cols[-1]]
    assert inner.mean() > 0.999
