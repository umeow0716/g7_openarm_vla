from __future__ import annotations

import numpy as np
import numpy.typing as npt

from g7_openarm_idl import VRJoy


def vr_joy_to_body_command(joy: VRJoy) -> npt.NDArray[np.float64]:
    """Map normalized VR axes to body-frame [vx, vy, wz].

    Robot body axes are +x forward, +y left, +yaw counter-clockwise.
    The VR stick axes are +x right and +y up, therefore:
      ly -> +vx, lx -> -vy, rx -> -wz.
    """

    axes = np.clip(
        np.array([joy.ly * 0.5, -joy.lx * 0.5, -joy.rx * 0.5], dtype=np.float64),
        -0.3,
        0.3,
    )
    return axes
