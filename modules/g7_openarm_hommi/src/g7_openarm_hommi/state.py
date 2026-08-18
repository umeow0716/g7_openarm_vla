from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from hommi_train.dataset import pose7_wxyz_to_matrix
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

from g7_openarm_idl import Odom
from g7_openarm_pinnzoo import PinnZooModel, kinematics
from g7_openarm_utils import (
    LEFT_GRIPPER_MOTOR_NAME,
    RIGHT_GRIPPER_MOTOR_NAME,
    motor_index,
)

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class RobotSnapshot:
    timestamp: float
    left_pose7: FloatArray
    right_pose7: FloatArray
    left_matrix: FloatArray
    right_matrix: FloatArray
    left_gripper_openness: float
    right_gripper_openness: float

    def pose7(self, side: str) -> FloatArray:
        if side == "left":
            return self.left_pose7
        if side == "right":
            return self.right_pose7
        raise ValueError(f"unknown arm side {side!r}")

    def matrix(self, side: str) -> FloatArray:
        if side == "left":
            return self.left_matrix
        if side == "right":
            return self.right_matrix
        raise ValueError(f"unknown arm side {side!r}")

    def gripper_openness(self, side: str) -> float:
        if side == "left":
            return self.left_gripper_openness
        if side == "right":
            return self.right_gripper_openness
        raise ValueError(f"unknown arm side {side!r}")


class RobotStateProjector:
    """Convert the latest lowstate/odom pair to HoMMI observation state via FK."""

    def __init__(self) -> None:
        self._model = PinnZooModel()

    def snapshot(
        self,
        lowstate: LowState_,
        odom: Odom,
        *,
        timestamp: float,
    ) -> RobotSnapshot:
        x_lib = self._model.build_x_lib(lowstate, odom)
        kin = np.asarray(kinematics(self._model, x_lib), dtype=np.float64)
        if kin.shape != (self._model.kinematics_size,):
            raise RuntimeError(
                f"PinnZoo kinematics returned shape {kin.shape}, "
                f"expected ({self._model.kinematics_size},)"
            )

        left_pose7 = kin[self._model.kinematics_pose_slice("L_tcp")].astype(
            np.float32, copy=True
        )
        right_pose7 = kin[self._model.kinematics_pose_slice("R_tcp")].astype(
            np.float32, copy=True
        )
        left_matrix = pose7_wxyz_to_matrix(left_pose7).astype(np.float32, copy=False)
        right_matrix = pose7_wxyz_to_matrix(right_pose7).astype(np.float32, copy=False)

        left_open = float(
            np.clip(
                lowstate.motor_state[motor_index(LEFT_GRIPPER_MOTOR_NAME)].q,
                0.0,
                1.0,
            )
        )
        right_open = float(
            np.clip(
                lowstate.motor_state[motor_index(RIGHT_GRIPPER_MOTOR_NAME)].q,
                0.0,
                1.0,
            )
        )

        return RobotSnapshot(
            timestamp=float(timestamp),
            left_pose7=left_pose7,
            right_pose7=right_pose7,
            left_matrix=left_matrix,
            right_matrix=right_matrix,
            left_gripper_openness=left_open,
            right_gripper_openness=right_open,
        )
