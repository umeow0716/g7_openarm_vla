from __future__ import annotations

import mujoco
import numpy as np

from g7_openarm_idl import EETarget

from .idl import array_to_pose


def load_hand_default_pose(model_path: str) -> EETarget:
    """Load the default left/right TCP poses from a MuJoCo XML model."""
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    left_hand = data.body("L_tcp")
    right_hand = data.body("R_tcp")

    left_pose = np.concatenate((left_hand.xpos, left_hand.xquat), dtype=np.float64)
    right_pose = np.concatenate((right_hand.xpos, right_hand.xquat), dtype=np.float64)
    return EETarget(array_to_pose(left_pose), array_to_pose(right_pose), 0.0, 0.0)
