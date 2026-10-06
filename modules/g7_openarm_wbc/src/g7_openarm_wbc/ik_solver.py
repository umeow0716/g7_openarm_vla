from __future__ import annotations

import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from g7_openarm_idl import pose_to_array
from g7_openarm_pinnzoo import (
    PinnZooModel,
    build_lib_x,
    kinematics,
    kinematics_jacobian,
    pinnzoo_q_idx,
    pinnzoo_dim_idx,
)
from g7_openarm_utils.layout import (
    Dim,
    Joint,
    KinBody,
    BASE_ENABLED,
    LEFT_ARM_ENABLED,
    RIGHT_ARM_ENABLED,
    BASE_JOINT,
    LEFT_ARM_JOINT,
    RIGHT_ARM_JOINT,
)
from typing import TYPE_CHECKING

from .utils import (
    ori_err_quat,
    quat_jac_to_ori_err_jac,
)

if TYPE_CHECKING:
    from g7_openarm_idl import EETarget, Odom
    from unitree_sdk2py.idl.default import HGLowState_


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


class G7OpenArmLQR():
    def __init__(self):
        self.model = PinnZooModel()
        
        self.kin_slice: dict[KinBody, slice] = {}

        for i, name in enumerate(self.model.kinematics_bodies):
            kin_body = KinBody.name_to_kin_body(name)
            start_idx = i * self.model.kinematics_body_size
            end_idx = start_idx + self.model.kinematics_body_size
            self.kin_slice[kin_body] = slice(start_idx, end_idx)

        assert self.kin_slice[KinBody.L_TCP] is not None
        assert self.kin_slice[KinBody.R_TCP] is not None

        self.left_arm_indices = [
            pinnzoo_q_idx(joint) for joint in LEFT_ARM_JOINT
        ]
        self.right_arm_indices = [
            pinnzoo_q_idx(joint) for joint in RIGHT_ARM_JOINT
        ]
        self.pinnzoo_quat_indices = [
            pinnzoo_dim_idx(dim) for dim in [Dim.QUAT_W, Dim.QUAT_X, Dim.QUAT_Y, Dim.QUAT_Z]
        ]

        self.damping = 1e-4

        self.Q_hand_pos = 200.0
        self.Q_hand_ori = 10.0

        self.nu: int = len(LEFT_ARM_JOINT) + len(RIGHT_ARM_JOINT)
        if BASE_ENABLED:
            self.nu += len(BASE_JOINT)

        arm_r_pos = np.array(
            [0.03, 0.03, 0.05, 0.10, 0.30, 0.50, 0.80],
            dtype=np.float64,
        )
        arm_r_ori = np.array(
            [0.80, 0.50, 0.30, 0.10, 0.05, 0.03, 0.03],
            dtype=np.float64,
        )

        assert len(LEFT_ARM_JOINT) == len(arm_r_pos)
        assert len(RIGHT_ARM_JOINT) == len(arm_r_pos)

        r_pos = np.concatenate([arm_r_pos, arm_r_pos])
        r_ori = np.concatenate([arm_r_ori, arm_r_ori])

        if BASE_ENABLED:
            r_pos = np.concatenate([
                np.array([2.5, 2.5, 2.0], dtype=np.float64),
                r_pos,
            ])
            r_ori = np.concatenate([
                np.array([10.0, 10.0, 1.0], dtype=np.float64),
                r_ori,
            ])

        self.R_pos = np.diag(r_pos)
        self.R_ori = np.diag(r_ori)
        
        self.u_base_slice = slice(0, 3)
        self.u_left_slice  = slice(3, 3+len(LEFT_ARM_JOINT))    if BASE_ENABLED else slice(0, len(LEFT_ARM_JOINT))
        self.u_right_slice = slice(3+len(LEFT_ARM_JOINT), None) if BASE_ENABLED else slice(len(LEFT_ARM_JOINT), None)

    def task_state_from_x_lib(
        self,
        lib_x: npt.NDArray[np.float64],
    ):
        assert self.model.kinematics_body_size == 7

        kin = kinematics(self.model, lib_x)
        left_pose  = kin[self.kin_slice[KinBody.L_TCP]]
        right_pose = kin[self.kin_slice[KinBody.R_TCP]]
        return TaskState(
            left_pose=PoseState(
                pos=left_pose[:3],
                quat=left_pose[3:]
            ),
            right_pose=PoseState(
                pos=right_pose[:3],
                quat=right_pose[3:]
            ),
        )

    def task_state_from_target(
        self,
        eetarget: "EETarget",
    ):
        left_pose  = np.array(pose_to_array(eetarget.left_target), dtype=np.float64)
        right_pose = np.array(pose_to_array(eetarget.right_target), dtype=np.float64)
        return TaskState(
            left_pose=PoseState(
                pos=left_pose[:3],
                quat=left_pose[3:]
            ),
            right_pose=PoseState(
                pos=right_pose[:3],
                quat=right_pose[3:]
            ),
        )

    def task_evaluate(
        self,
        state: TaskState,
        target: TaskState,
    ) -> TaskEvaluation:
        left_pos_err = (
            target.left_pose.pos - state.left_pose.pos
            if LEFT_ARM_ENABLED
            else np.zeros(3, dtype=np.float64)
        )
        left_ori_err = (
            ori_err_quat(state.left_pose.quat, target.left_pose.quat)
            if LEFT_ARM_ENABLED
            else np.zeros(3, dtype=np.float64)
        )
        right_pos_err = (
            target.right_pose.pos - state.right_pose.pos
            if RIGHT_ARM_ENABLED
            else np.zeros(3, dtype=np.float64)
        )
        right_ori_err = (
            ori_err_quat(state.right_pose.quat, target.right_pose.quat)
            if RIGHT_ARM_ENABLED
            else np.zeros(3, dtype=np.float64)
        )

        return TaskEvaluation(
            left_pos_err=left_pos_err,
            left_ori_err=left_ori_err,
            right_pos_err=right_pos_err,
            right_ori_err=right_ori_err,
        )

    def task_kinematic_jacobian(
        self,
        lib_x: npt.NDArray[np.float64],
    ):
        Jkin = kinematics_jacobian(self.model, lib_x)

        J_left_arm  = Jkin[:, self.left_arm_indices]
        J_right_arm = Jkin[:, self.right_arm_indices]

        if not BASE_ENABLED:
            return np.concatenate([J_left_arm, J_right_arm], axis=1)

        qw, qx, qy, qz = lib_x[self.pinnzoo_quat_indices]

        yaw = float(
            np.arctan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
        )

        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        x_idx = pinnzoo_dim_idx(Dim.X)
        y_idx = pinnzoo_dim_idx(Dim.Y)
        J_vx_body =  c * Jkin[:, [x_idx]] + s * Jkin[:, [y_idx]]
        J_vy_body = -s * Jkin[:, [x_idx]] + c * Jkin[:, [y_idx]]

        dq_dwz = 0.5 * np.array([-qz, qy, -qx, qw,], dtype=np.float64,)
        J_wz_body = (
            Jkin[:, self.pinnzoo_quat_indices] @ dq_dwz
        )[:, None]

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

    def state_cost(
        self,
        task_evaluation: TaskEvaluation,
    ) -> float:
        cost = 0.0
        if LEFT_ARM_ENABLED:
            cost += 0.5 * self.Q_hand_pos * (
                task_evaluation.left_pos_err @ task_evaluation.left_pos_err
            )
            cost += 0.5 * self.Q_hand_ori * (
                task_evaluation.left_ori_err @ task_evaluation.left_ori_err
            )

        if RIGHT_ARM_ENABLED:
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
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        l = self.state_cost(task_evaluation)

        lx_pos = np.zeros(self.nu, dtype=np.float64)
        lxx_pos = np.zeros((self.nu, self.nu), dtype=np.float64)

        lx_ori = np.zeros(self.nu, dtype=np.float64)
        lxx_ori = np.zeros((self.nu, self.nu), dtype=np.float64)

        if LEFT_ARM_ENABLED:
            J_left = Jkin[self.kin_slice[KinBody.L_TCP], :]
            Jp_left = J_left[:3, :]
            Jq_left = J_left[3:, :]
            _, Jr_left = quat_jac_to_ori_err_jac(
                Jq=Jq_left,
                state_quat=state.left_pose.quat,
                target_quat=target.left_pose.quat,
            )
            Je_left_pos = -Jp_left

            lx_pos += self.Q_hand_pos * (
                Je_left_pos.T @ task_evaluation.left_pos_err
            )
            lxx_pos += self.Q_hand_pos * (
                Je_left_pos.T @ Je_left_pos
            )

            lx_ori += self.Q_hand_ori * (
                Jr_left.T @ task_evaluation.left_ori_err
            )
            lxx_ori += self.Q_hand_ori * (
                Jr_left.T @ Jr_left
            )

        if RIGHT_ARM_ENABLED:
            J_right = Jkin[self.kin_slice[KinBody.R_TCP], :]
            Jp_right = J_right[:3, :]
            Jq_right = J_right[3:, :]
            _, Jr_right = quat_jac_to_ori_err_jac(
                Jq=Jq_right,
                state_quat=state.right_pose.quat,
                target_quat=target.right_pose.quat,
            )
            Je_right_pos = -Jp_right

            lx_pos += self.Q_hand_pos * (
                Je_right_pos.T @ task_evaluation.right_pos_err
            )
            lxx_pos += self.Q_hand_pos * (
                Je_right_pos.T @ Je_right_pos
            )

            lx_ori += self.Q_hand_ori * (
                Jr_right.T @ task_evaluation.right_ori_err
            )
            lxx_ori += self.Q_hand_ori * (
                Jr_right.T @ Jr_right
            )

        return l, lx_pos, lxx_pos, lx_ori, lxx_ori

    def solve_once(
        self,
        lowstate: "HGLowState_",
        odom: "Odom",
        eetarget: "EETarget",
    ):
        lib_x = build_lib_x(self.model, lowstate, odom)

        state = self.task_state_from_x_lib(lib_x)
        target = self.task_state_from_target(eetarget)
        
        task_eval = self.task_evaluate(state, target)

        Jkin = self.task_kinematic_jacobian(lib_x)
        _, lx_pos, lxx_pos, lx_ori, lxx_ori = self.state_cost_deriv(
            state=state,
            target=target,
            task_evaluation=task_eval,
            Jkin=Jkin,
        )

        H_pos = 0.01 * lxx_pos + self.R_pos
        g_pos = 0.1 * lx_pos
        H_pos += self.damping * np.eye(self.nu)
        u_pos = -np.linalg.solve(H_pos, g_pos)

        H_ori = 0.01 * lxx_ori + self.R_ori
        g_ori = 0.1 * lx_ori
        H_ori += self.damping * np.eye(self.nu)
        u_ori = -np.linalg.solve(H_ori, g_ori)

        u = u_pos + u_ori
        if not LEFT_ARM_ENABLED:
            u[self.u_left_slice]  = 0.0
        if not RIGHT_ARM_ENABLED:
            u[self.u_right_slice] = 0.0
        if not BASE_ENABLED:
            u_cache = u
            u = np.zeros((self.nu+3,), dtype=np.float64)
            u[3:] = u_cache
        else:
            self.prev_u_base = u[:3].copy()
        return u