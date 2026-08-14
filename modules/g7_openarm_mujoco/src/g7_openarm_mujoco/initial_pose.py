from __future__ import annotations

import mujoco
import numpy as np
import numpy.typing as npt


def hand_poses_for_arm_position(
    model: mujoco.MjModel,
    arm_position_7: tuple[float, ...],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return L/R TCP poses for the same seven-joint arm configuration."""
    if len(arm_position_7) != 7:
        raise ValueError(f"arm_position_7 must contain 7 values, got {len(arm_position_7)}")

    data = mujoco.MjData(model)
    for side in ("L", "R"):
        for joint_number, position in enumerate(arm_position_7, start=1):
            joint = model.joint(f"{side}_{joint_number}_joint")
            qpos_address = model.jnt_qposadr[joint.id]
            data.qpos[qpos_address] = position

    mujoco.mj_forward(model, data)

    left_hand = data.body("L_gripper_tcp_link")
    right_hand = data.body("R_gripper_tcp_link")
    left_pose = np.concatenate([left_hand.xpos.copy(), left_hand.xquat.copy()])
    right_pose = np.concatenate([right_hand.xpos.copy(), right_hand.xquat.copy()])
    return left_pose, right_pose
