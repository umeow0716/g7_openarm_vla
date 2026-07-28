from __future__ import annotations

import numpy as np

from g7_openarm_idl import VRJoy, VRJoy_default
from g7_openarm_wbc.vr_joy_control import vr_joy_to_body_command


def test_vrjoy_default_is_zero() -> None:
    joy = VRJoy_default()
    assert (joy.lx, joy.ly, joy.rx, joy.ry) == (0.0, 0.0, 0.0, 0.0)


def test_vrjoy_maps_to_body_frame_command() -> None:
    joy = VRJoy(lx=0.25, ly=0.75, rx=0.5, ry=-1.0)
    np.testing.assert_allclose(
        vr_joy_to_body_command(joy),
        np.array([0.75, -0.25, -0.5]),
    )


def test_vrjoy_command_is_clamped() -> None:
    joy = VRJoy(lx=2.0, ly=-2.0, rx=-3.0, ry=0.0)
    np.testing.assert_allclose(
        vr_joy_to_body_command(joy),
        np.array([-1.0, -1.0, 1.0]),
    )
