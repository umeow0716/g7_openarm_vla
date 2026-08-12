from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from g7_openarm_config import general_config
from g7_openarm_pinnzoo import PinnZooModel, kinematics, kinematics_jacobian
from g7_openarm_utils import (
    ARM_LOWSTATE_MOTOR_INDICES,
    ARM_VELOCITY_LIMIT_RAD_S,
    position_limited_velocity_bounds,
)

from .config import config
from .control_layout import control_size, tracked_arms, zero_inactive_arm_controls
from .utils import ori_err_quat, quat_jac_to_ori_err_jac

if TYPE_CHECKING:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    from g7_openarm_idl import EETarget, Odom


@dataclass
class PoseState:
    pos: npt.NDArray[np.float64]
    quat: npt.NDArray[np.float64]


@dataclass
class TaskState:
    left_pose: PoseState
    right_pose: PoseState


@dataclass
class TaskEvaluation:
    left_pos_err: npt.NDArray[np.float64]
    left_ori_err: npt.NDArray[np.float64]
    right_pos_err: npt.NDArray[np.float64]
    right_ori_err: npt.NDArray[np.float64]


def task_kinematic_jacobian(
    model: PinnZooModel,
    x_lib: npt.NDArray[np.float64],
    *,
    base_enabled: bool,
) -> npt.NDArray[np.float64]:
    Jkin = kinematics_jacobian(model, x_lib)

    J_left_arm = Jkin[:, 15:22]
    J_right_arm = Jkin[:, 24:31]

    if not base_enabled:
        return np.concatenate([J_left_arm, J_right_arm], axis=1)

    qw, qx, qy, qz = x_lib[3:7]

    yaw = float(
        np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
    )

    c = float(np.cos(yaw))
    s = float(np.sin(yaw))

    J_vx_body = c * Jkin[:, 0:1] + s * Jkin[:, 1:2]
    J_vy_body = -s * Jkin[:, 0:1] + c * Jkin[:, 1:2]

    dq_dwz = 0.5 * np.array(
        [
            -qz,
            qy,
            -qx,
            qw,
        ],
        dtype=np.float64,
    )

    J_wz_body = (Jkin[:, 3:7] @ dq_dwz)[:, None]

    return np.concatenate(
        [
            J_vx_body,
            J_vy_body,
            J_wz_body,
            J_left_arm,
            J_right_arm,
        ],
        axis=1,
    )


class G7OpenArmIKSolver:
    def __init__(
        self,
        lib_path: str | None = None,
        *,
        base_enabled: bool | None = None,
    ) -> None:
        self.model = PinnZooModel(lib_path)
        self.control_mode = general_config.control_mode
        self.base_enabled = general_config.base_enabled if base_enabled is None else base_enabled
        self.track_left, self.track_right = tracked_arms(self.control_mode)
        self.nu = control_size(base_enabled=self.base_enabled)

        self.Q_hand_pos = 200.0
        self.Q_hand_ori = 0.5
        self.R_du_base = np.diag(
            [
                8.0,
                8.0,
                1.0,
            ]
        ).astype(np.float64)

        self.prev_u_base = np.zeros(3, dtype=np.float64)

        u_max_full = np.array(
            [
                0.5,
                0.5,
                0.5,  # base vx, vy, omega
                # left arm: J1~J7
                1.57,
                1.57,  # J1, J2: DM-J8009P
                3.14,
                3.14,  # J3, J4: DM-J4340P / DM-J4340
                12.6,
                12.6,
                12.6,  # J5, J6, J7: DM-J4310
                # right arm: J1~J7
                1.57,
                1.57,  # J1, J2: DM-J8009P
                3.14,
                3.14,  # J3, J4: DM-J4340P / DM-J4340
                12.6,
                12.6,
                12.6,  # J5, J6, J7: DM-J4310
            ],
            dtype=np.float64,
        )
        u_max_full[3:] = np.minimum(u_max_full[3:], ARM_VELOCITY_LIMIT_RAD_S)
        self.u_max = u_max_full if self.base_enabled else u_max_full[3:]

        self.damping = 1e-4

        R_u_full = np.diag(
            [
                2.5,
                2.5,
                0.1,  # base vx, vy, omega
                0.05,
                0.05,  # left J1, J2: shoulder, 8009
                0.08,
                0.08,  # left J3, J4: 4340
                0.03,
                0.03,
                0.03,  # left J5, J6, J7: wrist, 4310
                0.05,
                0.05,  # right J1, J2
                0.08,
                0.08,  # right J3, J4
                0.03,
                0.03,
                0.03,  # right J5, J6, J7
            ]
        ).astype(np.float64)
        self.R_u = R_u_full if self.base_enabled else R_u_full[3:, 3:]

    def task_evaluate(
        self,
        state: TaskState,
        target: TaskState,
    ) -> TaskEvaluation:
        left_pos_err = (
            target.left_pose.pos - state.left_pose.pos
            if self.track_left
            else np.zeros(3, dtype=np.float64)
        )
        left_ori_err = (
            ori_err_quat(state.left_pose.quat, target.left_pose.quat)
            if self.track_left
            else np.zeros(3, dtype=np.float64)
        )
        right_pos_err = (
            target.right_pose.pos - state.right_pose.pos
            if self.track_right
            else np.zeros(3, dtype=np.float64)
        )
        right_ori_err = (
            ori_err_quat(state.right_pose.quat, target.right_pose.quat)
            if self.track_right
            else np.zeros(3, dtype=np.float64)
        )

        return TaskEvaluation(
            left_pos_err=left_pos_err,
            left_ori_err=left_ori_err,
            right_pos_err=right_pos_err,
            right_ori_err=right_ori_err,
        )

    def state_cost(
        self,
        task_evaluation: TaskEvaluation,
    ) -> float:
        cost = 0.0
        if self.track_left:
            cost += 0.5 * self.Q_hand_pos * (
                task_evaluation.left_pos_err @ task_evaluation.left_pos_err
            )
            cost += 0.5 * self.Q_hand_ori * (
                task_evaluation.left_ori_err @ task_evaluation.left_ori_err
            )

        if self.track_right:
            cost += 0.5 * self.Q_hand_pos * (
                task_evaluation.right_pos_err @ task_evaluation.right_pos_err
            )
            cost += 0.5 * self.Q_hand_ori * (
                task_evaluation.right_ori_err @ task_evaluation.right_ori_err
            )

        return float(cost)

    def state_cost_deriv(
        self,
        state: TaskState,
        target: TaskState,
        task_evaluation: TaskEvaluation,
        Jkin: npt.NDArray[np.float64],
    ) -> tuple[
        float,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        l = self.state_cost(task_evaluation)

        lx = np.zeros(self.nu, dtype=np.float64)
        lxx = np.zeros((self.nu, self.nu), dtype=np.float64)

        if self.track_left:
            Jp_left = Jkin[0:3, :]
            Jq_left = Jkin[3:7, :]
            _, Jr_left = quat_jac_to_ori_err_jac(
                Jq=Jq_left,
                state_quat=state.left_pose.quat,
                target_quat=target.left_pose.quat,
            )
            Je_left_pos = -Jp_left

            lx += self.Q_hand_pos * (Je_left_pos.T @ task_evaluation.left_pos_err)
            lxx += self.Q_hand_pos * (Je_left_pos.T @ Je_left_pos)

            lx += self.Q_hand_ori * (Jr_left.T @ task_evaluation.left_ori_err)
            lxx += self.Q_hand_ori * (Jr_left.T @ Jr_left)

        if self.track_right:
            Jp_right = Jkin[7:10, :]
            Jq_right = Jkin[10:14, :]
            _, Jr_right = quat_jac_to_ori_err_jac(
                Jq=Jq_right,
                state_quat=state.right_pose.quat,
                target_quat=target.right_pose.quat,
            )
            Je_right_pos = -Jp_right

            lx += self.Q_hand_pos * (Je_right_pos.T @ task_evaluation.right_pos_err)
            lxx += self.Q_hand_pos * (Je_right_pos.T @ Je_right_pos)

            lx += self.Q_hand_ori * (Jr_right.T @ task_evaluation.right_ori_err)
            lxx += self.Q_hand_ori * (Jr_right.T @ Jr_right)

        return l, lx, lxx

    def task_state_from_x_lib(self, x_lib: npt.NDArray[np.float64]):
        kin = kinematics(self.model, x_lib)
        return TaskState(
            left_pose=PoseState(pos=kin[:3], quat=kin[3:7]),
            right_pose=PoseState(pos=kin[7:10], quat=kin[10:14]),
        )

    def task_state_from_target(self, ee_target: "EETarget"):
        left_pos = ee_target.left_target.position
        left_quat = ee_target.left_target.orientation
        right_pos = ee_target.right_target.position
        right_quat = ee_target.right_target.orientation
        return TaskState(
            left_pose=PoseState(
                pos=np.array([left_pos.x, left_pos.y, left_pos.z], dtype=np.float64),
                quat=np.array(
                    [left_quat.w, left_quat.x, left_quat.y, left_quat.z], dtype=np.float64
                ),
            ),
            right_pose=PoseState(
                pos=np.array([right_pos.x, right_pos.y, right_pos.z], dtype=np.float64),
                quat=np.array(
                    [right_quat.w, right_quat.x, right_quat.y, right_quat.z], dtype=np.float64
                ),
            ),
        )

    def solve_once(
        self,
        lowstate: "LowState_",
        odom: "Odom",
        ee_target: "EETarget",
    ) -> npt.NDArray[np.float64]:
        x_lib = PinnZooModel.build_x_lib(lowstate, odom)

        state = self.task_state_from_x_lib(x_lib)
        task_target = self.task_state_from_target(ee_target)

        task_eval = self.task_evaluate(state, task_target)

        Jkin = task_kinematic_jacobian(
            self.model,
            x_lib,
            base_enabled=self.base_enabled,
        )

        _, lx, lxx = self.state_cost_deriv(
            state=state,
            target=task_target,
            task_evaluation=task_eval,
            Jkin=Jkin,
        )

        H = 0.01 * lxx + self.R_u
        g = 0.1 * lx

        if self.base_enabled:
            H[:3, :3] += self.R_du_base
            g[:3] -= self.R_du_base @ self.prev_u_base

        H = H + self.damping * np.eye(self.nu)

        u = -np.linalg.solve(H, g)

        u = np.clip(u, -self.u_max, self.u_max)
        zero_inactive_arm_controls(
            u,
            base_enabled=self.base_enabled,
            track_left=self.track_left,
            track_right=self.track_right,
        )

        arm_slice = slice(3, None) if self.base_enabled else slice(None)
        arm_position = np.array(
            [lowstate.motor_state[i].q for i in ARM_LOWSTATE_MOTOR_INDICES],
            dtype=np.float64,
        )
        arm_velocity_lower, arm_velocity_upper = position_limited_velocity_bounds(
            arm_position,
            self.u_max[arm_slice],
            config.interval,
        )
        u[arm_slice] = np.clip(
            u[arm_slice],
            arm_velocity_lower,
            arm_velocity_upper,
        )

        if self.base_enabled:
            self.prev_u_base = u[:3].copy()

        return u
