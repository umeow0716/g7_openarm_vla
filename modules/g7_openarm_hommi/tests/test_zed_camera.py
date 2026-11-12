from __future__ import annotations

import numpy as np
import pytest

from g7_openarm_hommi.zed_camera import (
    Rectifier,
    ZedCalibration,
    disparity_to_depth,
    load_zed_conf,
    remove_invisible,
)

CONF = """
[LEFT_CAM_HD]
fx=530.0
fy=530.0
cx=640.0
cy=360.0
k1=0.01
k2=0.0
p1=0.0
p2=0.0
k3=0.0

[RIGHT_CAM_HD]
fx=531.0
fy=531.0
cx=641.0
cy=359.0
k1=0.0
k2=0.0
p1=0.0
p2=0.0
k3=0.0

[STEREO]
Baseline=120.0
TY=0.0
TZ=0.0
RX_HD=0.0
CV_HD=0.0
RZ_HD=0.0
"""


def _calib() -> ZedCalibration:
    K = np.array([[530.0, 0, 640.0], [0, 530.0, 360.0], [0, 0, 1]])
    return ZedCalibration(
        image_size=(1280, 720),
        K1=K,
        D1=np.zeros(5),
        K2=K.copy(),
        D2=np.zeros(5),
        R=np.eye(3),
        T=np.array([[-0.12], [0.0], [0.0]]),
    )


def test_load_zed_conf_reads_intrinsics_and_baseline(tmp_path):
    path = tmp_path / "SN1234.conf"
    path.write_text(CONF)
    calib = load_zed_conf(path, (1280, 720))
    assert calib.K1[0, 0] == pytest.approx(530.0)
    assert calib.K2[1, 2] == pytest.approx(359.0)
    assert calib.D1[0] == pytest.approx(0.01)
    assert calib.baseline == pytest.approx(0.12)


def test_load_zed_conf_rejects_unknown_eye_width(tmp_path):
    path = tmp_path / "x.conf"
    path.write_text(CONF)
    with pytest.raises(ValueError):
        load_zed_conf(path, (999, 720))


def test_rectifier_scales_to_working_size_and_keeps_baseline():
    rect = Rectifier(_calib(), (640, 360))
    assert rect.size == (640, 360)
    assert rect.baseline == pytest.approx(0.12, rel=1e-6)
    assert rect.K[0, 0] == pytest.approx(265.0, rel=1e-3)
    assert rect.K[1, 2] == pytest.approx(180.0, rel=1e-3)


def test_rectifier_rejects_flipped_eye_order():
    calib = _calib()
    flipped = ZedCalibration(
        calib.image_size, calib.K1, calib.D1, calib.K2, calib.D2, calib.R, -calib.T
    )
    with pytest.raises(RuntimeError):
        Rectifier(flipped, (640, 360))


def test_disparity_to_depth_inverts_z_equals_f_b_over_d():
    fx, baseline = 265.0, 0.12
    z_true = np.array([[0.5, 1.0, 2.0]], dtype=np.float32)
    disp = (fx * baseline / z_true).astype(np.float32)
    z = disparity_to_depth(disp, fx, baseline)
    np.testing.assert_allclose(z, z_true, rtol=1e-5)


def test_disparity_to_depth_marks_invalid_and_far_as_zero():
    fx, baseline = 265.0, 0.12
    disp = np.array([[0.0, np.nan, 1e-6, 10.0]], dtype=np.float32)
    z = disparity_to_depth(disp, fx, baseline, zfar=100.0)
    assert z[0, 0] == 0.0 and z[0, 1] == 0.0 and z[0, 2] == 0.0
    assert z[0, 3] == pytest.approx(fx * baseline / 10.0, rel=1e-5)


def test_remove_invisible_drops_left_pixels_without_right_match():
    disp = np.full((1, 8), 3.0, dtype=np.float32)
    out = remove_invisible(disp)
    assert (out[0, :3] == 0).all()
    assert (out[0, 3:] == 3.0).all()


def test_rectification_rotation_matches_rectifier(tmp_path):
    from g7_openarm_hommi.zed_camera import rectification_rotation

    path = tmp_path / "SN1234.conf"
    path.write_text(CONF)
    r1 = rectification_rotation(path)
    np.testing.assert_allclose(r1 @ r1.T, np.eye(3), atol=1e-9)
    expected = Rectifier(load_zed_conf(path, (1280, 720)), (960, 576)).R1
    np.testing.assert_allclose(r1, expected)
