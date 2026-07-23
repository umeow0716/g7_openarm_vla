import platform
import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from g7_openarm_pinnzoo import PinnZooModel, kinematics, kinematics_jacobian

from .utils import ori_err_quat, quat_jac_to_ori_err_jac, yaw_to_quat_wxyz

if TYPE_CHECKING:
    from g7_openarm_idl import Odom, EETarget
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


class NotSupportedArchitecture(Exception):
    pass

def get_arch():
    machine = platform.machine().lower()
    
    if machine in ('x86_64', 'amd64'):
        return 'x86_64'
    elif machine in ('aarch64', 'arm64'):
        return 'aarch64'

    raise NotSupportedArchitecture(f'{platform.machine().lower()} is not support')


DEFAULT_LIB_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent / "include" / f"libg7_openarm_quat_{get_arch()}.so"


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
    left_pos_err:  npt.NDArray[np.float64]
    left_ori_err:  npt.NDArray[np.float64]
    right_pos_err: npt.NDArray[np.float64]
    right_ori_err: npt.NDArray[np.float64]

def task_kinematic_jacobian(
    model: PinnZooModel,
    x_lib: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    Jkin = kinematics_jacobian(model, x_lib)

    qw, qx, qy, qz = x_lib[3:7]

    yaw = float(np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    ))

    c = float(np.cos(yaw))
    s = float(np.sin(yaw))

    J_vx_body = c * Jkin[:, 0:1] + s * Jkin[:, 1:2]
    J_vy_body = -s * Jkin[:, 0:1] + c * Jkin[:, 1:2]

    dq_dwz = 0.5 * np.array([
        -qz,
         qy,
        -qx,
         qw,
    ], dtype=np.float64)

    J_wz_body = (Jkin[:, 3:7] @ dq_dwz)[:, None]

    J_left_arm = Jkin[:, 15:22]
    J_right_arm = Jkin[:, 24:31]

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

def odom_velocity_world_to_body(
    odom: "Odom",
) -> npt.NDArray[np.float64]:
    qw = odom.quaternion.w
    qx = odom.quaternion.x
    qy = odom.quaternion.y
    qz = odom.quaternion.z

    R_world_body = np.array([
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qw * qz),
            2.0 * (qx * qz + qw * qy),
        ],
        [
            2.0 * (qx * qy + qw * qz),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qw * qx),
        ],
        [
            2.0 * (qx * qz - qw * qy),
            2.0 * (qy * qz + qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ], dtype=np.float64)

    R_body_world = R_world_body.T

    velocity_world = np.array([
        odom.velocity.x,
        odom.velocity.y,
        odom.velocity.z,
    ], dtype=np.float64)

    angular_velocity_world = np.array([
        odom.angular_velocity.x,
        odom.angular_velocity.y,
        odom.angular_velocity.z,
    ], dtype=np.float64)

    velocity_body = R_body_world @ velocity_world
    angular_velocity_body = R_body_world @ angular_velocity_world

    return np.concatenate([
        velocity_body,
        angular_velocity_body,
    ])


class G7OpenArmIKSolver:
    def __init__(
        self,
        lib_path: str | None = None,
    ) -> None:
        self.lib_path = str(DEFAULT_LIB_PATH if lib_path is None else lib_path)
        self.model = PinnZooModel(self.lib_path)
        
        self.nx = 17
        
        self.Q_hand_pos = 200.0
        self.Q_hand_ori = 0.5
        
        self.R_du_base = np.diag([
            8.0, 8.0, 1.0,
        ]).astype(np.float64)
        
        self.prev_u_base = np.zeros(3, dtype=np.float64)
        
        self.u_max = np.array([
            0.5, 0.5, 0.5,          # base vx, vy, omega

            # left arm: J1~J7
            2.0, 2.0,               # J1, J2: DM-J8009P
            1.5, 1.5,               # J3, J4: DM-J4340P / DM-J4340
            3.0, 3.0, 3.0,          # J5, J6, J7: DM-J4310

            # right arm: J1~J7
            2.0, 2.0,               # J1, J2: DM-J8009P
            1.5, 1.5,               # J3, J4: DM-J4340P / DM-J4340
            3.0, 3.0, 3.0,          # J5, J6, J7: DM-J4310
        ], dtype=np.float64)
        
        self.damping = 1e-4

        self.R_u = np.diag([
            2.5, 2.5, 0.1,          # base vx, vy, omega

            0.05, 0.05,             # left J1, J2: shoulder, 8009
            0.08, 0.08,             # left J3, J4: 4340
            0.03, 0.03, 0.03,       # left J5, J6, J7: wrist, 4310

            0.05, 0.05,             # right J1, J2
            0.08, 0.08,             # right J3, J4
            0.03, 0.03, 0.03,       # right J5, J6, J7
        ]).astype(np.float64)
    
    def task_evaluate(
        self,
        state: TaskState,
        target: TaskState,
    ) -> TaskEvaluation:
        left_pos_err = target.left_pose.pos - state.left_pose.pos
        right_pos_err = target.right_pose.pos - state.right_pose.pos

        left_ori_err = ori_err_quat(state.left_pose.quat, target.left_pose.quat)
        right_ori_err = ori_err_quat(state.right_pose.quat, target.right_pose.quat)
        
        return TaskEvaluation(
            left_pos_err=left_pos_err,
            left_ori_err=left_ori_err,
            right_pos_err=right_pos_err,
            right_ori_err=right_ori_err,
        )
        
    def state_cost(
        self,
        task_evaluation: TaskEvaluation,
    ) -> npt.NDArray[np.float64]:
        return \
            0.5 * self.Q_hand_pos * (task_evaluation.left_pos_err  @ task_evaluation.left_pos_err) + \
            0.5 * self.Q_hand_ori * (task_evaluation.left_ori_err  @ task_evaluation.left_ori_err) + \
            0.5 * self.Q_hand_pos * (task_evaluation.right_pos_err @ task_evaluation.right_pos_err) + \
            0.5 * self.Q_hand_ori * (task_evaluation.right_ori_err @ task_evaluation.right_ori_err)

    def state_cost_deriv(
        self,
        state: TaskState,
        target: TaskState,
        task_evaluation: TaskEvaluation,
        Jkin: npt.NDArray[np.float64]
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        Jp_left  = Jkin[0:3,   :]
        Jq_left  = Jkin[3:7,   :]
        Jp_right = Jkin[7:10,  :]
        Jq_right = Jkin[10:14, :]

        _, Jr_left = quat_jac_to_ori_err_jac(
            Jq=Jq_left,
            state_quat=state.left_pose.quat,
            target_quat=target.left_pose.quat,
        )
        _, Jr_right = quat_jac_to_ori_err_jac(
            Jq=Jq_right,
            state_quat=state.right_pose.quat,
            target_quat=target.right_pose.quat,
        )

        Je_left_pos = -Jp_left
        Je_right_pos = -Jp_right
        Je_left_ori = Jr_left
        Je_right_ori = Jr_right

        l = self.state_cost(task_evaluation)

        lx = np.zeros(self.nx, dtype=np.float64)
        lxx = np.zeros((self.nx, self.nx), dtype=np.float64)
        
        lx += self.Q_hand_pos * (Je_left_pos.T @ task_evaluation.left_pos_err)
        lxx += self.Q_hand_pos * (Je_left_pos.T @ Je_left_pos)

        lx += self.Q_hand_ori * (Je_left_ori.T @ task_evaluation.left_ori_err)
        lxx += self.Q_hand_ori * (Je_left_ori.T @ Je_left_ori)

        lx += self.Q_hand_pos * (Je_right_pos.T @ task_evaluation.right_pos_err)
        lxx += self.Q_hand_pos * (Je_right_pos.T @ Je_right_pos)

        lx += self.Q_hand_ori * (Je_right_ori.T @ task_evaluation.right_ori_err)
        lxx += self.Q_hand_ori * (Je_right_ori.T @ Je_right_ori)

        return l, lx, lxx
    
    def build_x_lib(
        self,
        lowstate: "LowState_",
        odom: "Odom"
    ) -> npt.NDArray[np.float64]:
        motor_state = lowstate.motor_state
        
        position         = np.array([odom.position.x, odom.position.y, odom.position.z], dtype=np.float64)
        quat             = np.array([odom.quaternion.w, odom.quaternion.x, odom.quaternion.y, odom.quaternion.z], dtype=np.float64)

        q_0_14 = np.array([m.q for m in motor_state[0:15]], dtype=np.float64)
        q_16_22 = np.array([m.q for m in motor_state[16:23]], dtype=np.float64)

        dq_0_14 = np.array([m.dq for m in motor_state[0:15]], dtype=np.float64)
        dq_16_22 = np.array([m.dq for m in motor_state[16:23]], dtype=np.float64)

        return np.concatenate((
            position,
            quat,
            q_0_14,
            np.zeros(2, dtype=np.float64),
            q_16_22,
            np.zeros(2, dtype=np.float64),
            odom_velocity_world_to_body(odom),
            dq_0_14,
            np.zeros(2, dtype=np.float64),
            dq_16_22,
            np.zeros(2, dtype=np.float64),
        ))

    def task_state_from_x_lib(
        self,
        x_lib: npt.NDArray[np.float64]
    ):
        kin = kinematics(self.model, x_lib)
        return TaskState(
            left_pose=PoseState(
                pos=kin[:3],
                quat=kin[3:7]
            ),
            right_pose=PoseState(
                pos=kin[7:10],
                quat=kin[10:14]
            )
        )

    def task_state_from_target(
        self,
        ee_target: "EETarget"
    ):
        left_pos   = ee_target.left_target.position
        left_quat  = ee_target.left_target.orientation
        right_pos  = ee_target.right_target.position
        right_quat = ee_target.right_target.orientation
        return TaskState(
            left_pose=PoseState(
                pos=np.array([left_pos.x, left_pos.y, left_pos.z], dtype=np.float64),
                quat=np.array([left_quat.w, left_quat.x, left_quat.y, left_quat.z], dtype=np.float64),
            ),
            right_pose=PoseState(
                pos=np.array([right_pos.x, right_pos.y, right_pos.z], dtype=np.float64),
                quat=np.array([right_quat.w, right_quat.x, right_quat.y, right_quat.z], dtype=np.float64),
            ),
        )

    def solve_once(
        self,
        lowstate: "LowState_",
        odom: "Odom",
        ee_target: "EETarget",
    ) -> npt.NDArray[np.float64]:
        x_lib = self.build_x_lib(lowstate, odom)
        
        state  = self.task_state_from_x_lib(x_lib)
        task_target = self.task_state_from_target(ee_target)
        
        task_eval = self.task_evaluate(state, task_target)

        Jkin = task_kinematic_jacobian(self.model, x_lib)

        _, lx, lxx = self.state_cost_deriv(
            state=state,
            target=task_target,
            task_evaluation=task_eval,
            Jkin=Jkin,
        )

        H = 0.01 * lxx + self.R_u
        g = 0.1 * lx
        
        H[:3, :3] += self.R_du_base
        g[:3] -= self.R_du_base @ self.prev_u_base

        H = H + self.damping * np.eye(self.nx)

        u = -np.linalg.solve(H, g)

        u = np.clip(u, -self.u_max, self.u_max)
        
        self.prev_u_base = u[:3].copy()

        return u
