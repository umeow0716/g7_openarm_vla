import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from g7_openarm_utils.layout.control_layout import (
    Joint,
    ARM_JOINT,
    LEFT_ARM_ENABLED,
    RIGHT_ARM_ENABLED,
    low_idx,
)
from typing import TYPE_CHECKING

from .config import config

if TYPE_CHECKING:
    from unitree_sdk2py.idl.default import HGLowState_


JERK_MAX = np.tile(np.array([
    100.0, 100.0,
    150.0, 150.0,
    300.0, 300.0, 300.0,
]), 2)

TORQUE_MAX = np.tile(np.array([
    40.0, 40.0,
    27.0, 27.0,
     7.0,  7.0,  7.0,
]), 2)

ACCEL_MAX = np.tile(np.array([
    10.0, 10.0,
    15.0, 15.0,
    30.0, 30.0, 30.0,
]), 2)

VEL_MAX = np.tile(np.array([
    10.0, 10.0,
    3.5,  3.5,
    12.0, 12.0, 12.0,
]), 2)

POS_MAX = np.tile(np.array([
    3.49, 3.31,
    1.04,  0.0,
    1.57, 0.78, 1.57,
]), 2)

POS_MIN = np.tile(np.array([
    -1.39, -0.17,
    -0.52, -2.44,
    -1.57, -0.78, -1.57,
]), 2)

ACTIVE_ARM_MASK = np.array(
    [LEFT_ARM_ENABLED] * 7
    + [RIGHT_ARM_ENABLED] * 7,
    dtype=bool,
)


@dataclass
class ReferenceResult:
    q: dict["Joint", float]
    dq: dict["Joint", float]
    ddq: dict["Joint", float]


class ReferenceGenerator:
    def __init__(self):
        self.initialized = False
        self.started = False
        
        self.e_soft = 0.05
        self.e_hard = 0.10
        
        self.rho_tau = 0.8
        
        assert len(ARM_JOINT) == 14

        self.q_ref   = np.zeros((14,))
        self.q_meas  = np.zeros((14,)) 
        self.dq_ref  = np.zeros((14,))
        self.ddq_ref = np.zeros((14,))
        
        self.tau_raw = np.zeros((14,))
        
        self.gamma = 0.0
        
        self.delta_t = 1.0 / config.hz
        self.r_fall = 5.0
        self.r_rise = 0.5
        
        self.omega_c = 50.0
        self.omega_f = 0.5 * self.omega_c
        self.zeta_f  = 1.0

    def initialize(
        self,
        lowstate: "HGLowState_"
    ):
        for i, joint in enumerate(ARM_JOINT):
            state = lowstate.motor_state[low_idx(joint)]
            self.q_meas[i] = state.q
            self.q_ref[i] = state.q
            self.dq_ref[i] = state.dq
            self.ddq_ref[i] = 0.0
        self.initialized = True

    def generate(
        self,
        lowstate: "HGLowState_",
        dq_des: npt.NDArray[np.float64],
    ):
        assert self.initialized

        dq_des_active = np.where(ACTIVE_ARM_MASK, dq_des, 0.0)

        for i, joint in enumerate(ARM_JOINT):
            self.q_meas[i] = lowstate.motor_state[low_idx(joint)].q

        if not self.started:
            self.q_ref[:] = self.q_meas
            self.dq_ref[:] = 0.0
            self.ddq_ref[:] = 0.0
            self.started = True

        tracking_error = self.q_ref - self.q_meas
        outside_tracking_bound = np.abs(tracking_error) > self.e_hard
        self.q_ref = np.where(
            outside_tracking_bound,
            self.q_meas + np.clip(tracking_error, -self.e_soft, self.e_soft),
            self.q_ref,
        )

        tau_allow = self.rho_tau * TORQUE_MAX
        
        gamma_tau = tau_allow / np.maximum(
            np.abs(self.tau_raw),
            tau_allow,
        )
        
        gamma_all = gamma_tau

        r_gamma = np.where(
            gamma_all < self.gamma,
            self.r_fall,
            self.r_rise,
        )
        
        self.gamma += np.clip(
            gamma_all - self.gamma,
            -r_gamma * self.delta_t,
            r_gamma * self.delta_t
        )

        dq_goal = self.gamma * np.clip(dq_des_active, -VEL_MAX, VEL_MAX)

        upper_distance = np.maximum(POS_MAX - self.q_ref, 0.0)
        lower_distance = np.maximum(self.q_ref - POS_MIN, 0.0)

        upper_velocity_limit = np.sqrt(2.0 * ACCEL_MAX * upper_distance)
        lower_velocity_limit = np.sqrt(2.0 * ACCEL_MAX * lower_distance)
        
        dq_goal = np.minimum(dq_goal,  upper_velocity_limit)
        dq_goal = np.maximum(dq_goal, -lower_velocity_limit)

        j_cmd   = (self.omega_f ** 2) * (dq_goal - self.dq_ref) - 2 * self.zeta_f * self.omega_f * self.ddq_ref
        j_tilde = np.clip(j_cmd, -JERK_MAX, JERK_MAX)
        
        block_jerk = (
            ((self.ddq_ref >= ACCEL_MAX) & (j_tilde > 0))
            | ((self.ddq_ref <= -ACCEL_MAX) & (j_tilde < 0))
        )
                
        j_ref = np.where(block_jerk, 0.0, j_tilde)
        
        next_ddq_ref = np.clip(
            self.ddq_ref + j_ref * self.delta_t,
            -ACCEL_MAX,
            ACCEL_MAX
        )
        next_dq_ref = np.clip(
            self.dq_ref + ((self.ddq_ref + next_ddq_ref) / 2.0) * self.delta_t,
            -VEL_MAX,
            VEL_MAX,
        )
        next_q_ref = self.q_ref + ((self.dq_ref + next_dq_ref) / 2.0) * self.delta_t
        
        hit_upper = next_q_ref > POS_MAX
        hit_lower = next_q_ref < POS_MIN
        hit_limit = hit_upper | hit_lower

        next_q_ref = np.clip(next_q_ref, POS_MIN, POS_MAX)
        next_dq_ref = np.where(hit_limit, 0.0, next_dq_ref)
        next_ddq_ref = np.where(hit_limit, 0.0, next_ddq_ref)
        
        self.ddq_ref = next_ddq_ref
        self.dq_ref  = next_dq_ref
        self.q_ref   = next_q_ref
        
        q_result = {
            joint: self.q_ref[i]
            for i, joint in enumerate(ARM_JOINT)
        }
        dq_result = {
            joint: self.dq_ref[i]
            for i, joint in enumerate(ARM_JOINT)
            if not Joint.is_gripper(joint)
        }
        ddq_result = {
            joint: self.ddq_ref[i]
            for i, joint in enumerate(ARM_JOINT)
            if not Joint.is_gripper(joint)
        }

        return ReferenceResult(
            q=q_result,
            dq=dq_result,
            ddq=ddq_result,
        )

    def update_tau_raw(
        self,
        tau_raw: npt.NDArray[np.float64],
    ):
        tau_raw = np.asarray(tau_raw, dtype=np.float64)

        assert tau_raw.shape == (14,)

        self.tau_raw[:] = tau_raw
