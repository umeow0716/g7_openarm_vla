from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from g7_openarm_config import general_config
from g7_openarm_pinnzoo import PinnZooModel, kinematics, kinematics_jacobian
from g7_openarm_utils import (
    ARM_MOTOR_NAMES,
    ARM_VELOCITY_LIMIT_RAD_S,
    FLOATING_BASE_CONFIG_NAMES,
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    motor_state_values,
    position_limited_velocity_bounds,
)

from .config import config
from .control_layout import (
    ARM_CONTROL_NAMES,
    BASE_CONTROL_NAMES,
    control_indices,
    control_names,
    control_size,
    tracked_arms,
    zero_inactive_arm_controls,
)
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


LEFT_EE_BODY_NAME = "L_tcp"
RIGHT_EE_BODY_NAME = "R_tcp"


def _pose_component_slices(pose_slice: slice) -> tuple[slice, slice]:
    if (
        pose_slice.start is None
        or pose_slice.stop is None
        or pose_slice.stop - pose_slice.start != 7
    ):
        raise ValueError(f"Expected 7-element pose slice, got {pose_slice}")
    start = pose_slice.start
    return slice(start, start + 3), slice(start + 3, start + 7)


def task_kinematic_jacobian(
    model: PinnZooModel,
    x_lib: npt.NDArray[np.float64],
    *,
    base_enabled: bool,
) -> npt.NDArray[np.float64]:
    Jkin = kinematics_jacobian(model, x_lib)

    J_left_arm = Jkin[:, model.q_indices(LEFT_ARM_JOINT_NAMES)]
    J_right_arm = Jkin[:, model.q_indices(RIGHT_ARM_JOINT_NAMES)]

    if not base_enabled:
        return np.concatenate([J_left_arm, J_right_arm], axis=1)

    quaternion_names = FLOATING_BASE_CONFIG_NAMES[3:]
    qw, qx, qy, qz = x_lib[model.q_indices(quaternion_names)]

    yaw = float(
        np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
    )

    c = float(np.cos(yaw))
    s = float(np.sin(yaw))

    x_column = Jkin[:, model.q_indices(("x",))]
    y_column = Jkin[:, model.q_indices(("y",))]
    J_vx_body = c * x_column + s * y_column
    J_vy_body = -s * x_column + c * y_column

    dq_dwz = 0.5 * np.array(
        [
            -qz,
            qy,
            -qx,
            qw,
        ],
        dtype=np.float64,
    )

    J_wz_body = (Jkin[:, model.q_indices(quaternion_names)] @ dq_dwz)[:, None]

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
        self._left_pose_slice = self.model.kinematics_pose_slice(LEFT_EE_BODY_NAME)
        self._right_pose_slice = self.model.kinematics_pose_slice(RIGHT_EE_BODY_NAME)
        self._left_pos_slice, self._left_quat_slice = _pose_component_slices(
            self._left_pose_slice
        )
        self._right_pos_slice, self._right_quat_slice = _pose_component_slices(
            self._right_pose_slice
        )

        self._base_control_idx = control_indices(
            BASE_CONTROL_NAMES, base_enabled=self.base_enabled
        ) if self.base_enabled else np.empty(0, dtype=np.intp)
        self._arm_control_idx = control_indices(
            ARM_CONTROL_NAMES, base_enabled=self.base_enabled
        )

        self.R_du_base = np.diag([8.0, 8.0, 1.0]).astype(np.float64)
        self.prev_u_base = np.zeros(len(BASE_CONTROL_NAMES), dtype=np.float64)

        def arm_velocity_cap(name: str) -> float:
            joint_number = int(name.split("_")[1])
            if joint_number <= 2:
                return 1.57
            if joint_number <= 4:
                return 3.14
            return 12.6

        arm_velocity_cap_by_name = {
            name: arm_velocity_cap(name) for name in ARM_MOTOR_NAMES
        }
        control_velocity_cap_by_name = {
            "base_vx": 0.5,
            "base_vy": 0.5,
            "base_wz": 0.5,
            **arm_velocity_cap_by_name,
        }
        active_control_names = control_names(base_enabled=self.base_enabled)
        self.u_max = np.asarray(
            [control_velocity_cap_by_name[name] for name in active_control_names],
            dtype=np.float64,
        )
        self.u_max[self._arm_control_idx] = np.minimum(
            self.u_max[self._arm_control_idx], ARM_VELOCITY_LIMIT_RAD_S
        )

        self.damping = 1e-4

        def arm_regularization_weight(name: str) -> float:
            joint_number = int(name.split("_")[1])
            if joint_number <= 2:
                return 0.05
            if joint_number <= 4:
                return 0.08
            return 0.03

        arm_weight_by_name = {
            name: arm_regularization_weight(name) for name in ARM_MOTOR_NAMES
        }
        control_weight_by_name = {
            "base_vx": 2.5,
            "base_vy": 2.5,
            "base_wz": 0.1,
            **arm_weight_by_name,
        }
        self.R_u = np.diag(
            [control_weight_by_name[name] for name in active_control_names]
        ).astype(np.float64)

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
            Jp_left = Jkin[self._left_pos_slice, :]
            Jq_left = Jkin[self._left_quat_slice, :]
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
            Jp_right = Jkin[self._right_pos_slice, :]
            Jq_right = Jkin[self._right_quat_slice, :]
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
            left_pose=PoseState(
                pos=kin[self._left_pos_slice],
                quat=kin[self._left_quat_slice],
            ),
            right_pose=PoseState(
                pos=kin[self._right_pos_slice],
                quat=kin[self._right_quat_slice],
            ),
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
        x_lib = self.model.build_x_lib(lowstate, odom)

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
            H[np.ix_(self._base_control_idx, self._base_control_idx)] += self.R_du_base
            g[self._base_control_idx] -= self.R_du_base @ self.prev_u_base

        H = H + self.damping * np.eye(self.nu)

        u = -np.linalg.solve(H, g)

        u = np.clip(u, -self.u_max, self.u_max)
        zero_inactive_arm_controls(
            u,
            base_enabled=self.base_enabled,
            track_left=self.track_left,
            track_right=self.track_right,
        )

        arm_position = motor_state_values(lowstate, ARM_MOTOR_NAMES, "q")
        arm_velocity_lower, arm_velocity_upper = position_limited_velocity_bounds(
            arm_position,
            self.u_max[self._arm_control_idx],
            config.interval,
        )
        u[self._arm_control_idx] = np.clip(
            u[self._arm_control_idx],
            arm_velocity_lower,
            arm_velocity_upper,
        )

        if self.base_enabled:
            self.prev_u_base = u[self._base_control_idx].copy()

        return u
