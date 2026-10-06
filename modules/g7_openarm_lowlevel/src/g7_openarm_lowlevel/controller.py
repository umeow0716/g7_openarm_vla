from __future__ import annotations

import numpy as np
import numpy.typing as npt

from g7_openarm_pinnzoo import (
    PinnZooModel,
    build_lib_x,
    pinnzoo_vdot_idx,
    inverse_dynamics,
    mass_matrix,
)
from g7_openarm_utils.layout import (
    Joint,
    BASE_JOINT,
    ARM_JOINT,
    low_idx,
)
from typing import TYPE_CHECKING
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as HGLowCmd_default

from .reference_generator import ReferenceGenerator
from .wheel_kinematic import WheelKinematic

if TYPE_CHECKING:
    from unitree_sdk2py.idl.default import HGLowState_
    from g7_openarm_idl import Odom, WBCLowCmd


class Controller:
    def __init__(
        self,
        lib_path: str | None = None,
    ) -> None:
        self.model = PinnZooModel(lib_path)
        self.reference_generator = ReferenceGenerator()
        self.wheel_kinematic = WheelKinematic()

    def initialize(
        self,
        lowstate: "HGLowState_"
    ):
        self.reference_generator.initialize(lowstate)

    def compute_arm_kd(
        self,
        M_diag: npt.NDArray[np.float64],
        zeta=0.3,
        omega=2.0,
    ):
        Kd = 2.0 * zeta * omega * M_diag
        return {
            joint: Kd[pinnzoo_vdot_idx(joint)]
            for joint in ARM_JOINT
        }

    def compute_arm_kp(
        self,
        M_diag: npt.NDArray[np.float64],
        omega=2.0,
    ):
        Kp = (omega**2) * M_diag
        return {
            joint: Kp[pinnzoo_vdot_idx(joint)]
            for joint in ARM_JOINT
        }

    def update(
        self,
        lowstate: "HGLowState_",
        odom: Odom,
        wbclowcmd: "WBCLowCmd"
    ):
        vx, vy, wz = wbclowcmd.mobile[:3]
        best_wheel = self.wheel_kinematic.choose_best_pose(lowstate, vx, vy, wz)

        lib_x = build_lib_x(self.model, lowstate, odom)
        
        M = mass_matrix(self.model, lib_x)
        M_diag = np.diag(M)
        Kp = self.compute_arm_kp(M_diag)
        Kd = self.compute_arm_kd(M_diag)

        dq_des = np.array(wbclowcmd.dq_des, dtype=np.float64)
        ref = self.reference_generator.generate(lowstate, dq_des)

        vdot  = np.zeros((self.model.nv,), dtype=np.float64)
        for joint in ARM_JOINT:
            vdot[pinnzoo_vdot_idx(joint)] = ref.ddq[joint]
        vdot[0] = odom.vdot.x
        vdot[1] = odom.vdot.y
        vdot[2] = odom.vdot.z
        vdot[3] = odom.angular_vdot.x
        vdot[4] = odom.angular_vdot.y
        vdot[5] = odom.angular_vdot.z

        tau_ff = inverse_dynamics(self.model, lib_x, vdot)

        tau_raw = np.zeros((14,), dtype=np.float64)
        for i, joint in enumerate(ARM_JOINT):
            low_motor_state = lowstate.motor_state[low_idx(joint)]
            q_mea_i  = low_motor_state.q
            dq_mea_i = low_motor_state.dq
            tau_raw[i] = \
                Kp[joint] * (ref.q[joint]  - q_mea_i)  + \
                Kd[joint] * (ref.dq[joint] - dq_mea_i) + \
                tau_ff[pinnzoo_vdot_idx(joint)]
        self.reference_generator.update_tau_raw(tau_raw)

        result = HGLowCmd_default()
        
        for joint in BASE_JOINT:
            idx = low_idx(joint)
            if Joint.is_wheel(joint):
                result.motor_cmd[idx].kp  = 0.0
                result.motor_cmd[idx].kd  = 5.0
                result.motor_cmd[idx].q   = 0.0
                result.motor_cmd[idx].dq  = best_wheel[joint]
                result.motor_cmd[idx].tau = 0.0
            else:
                result.motor_cmd[idx].kp  = 100.0
                result.motor_cmd[idx].kd  = 1.0
                result.motor_cmd[idx].q   = best_wheel[joint]
                result.motor_cmd[idx].dq  = 0.0
                result.motor_cmd[idx].tau = 0.0
        
        for joint in ARM_JOINT:
            idx = low_idx(joint)
            result.motor_cmd[idx].kp  = Kp[joint]
            result.motor_cmd[idx].kd  = Kd[joint]
            result.motor_cmd[idx].q   = ref.q[joint]
            result.motor_cmd[idx].dq  = ref.dq[joint]
            result.motor_cmd[idx].tau = tau_ff[pinnzoo_vdot_idx(joint)]

        lli = low_idx(Joint.L8)
        rli = low_idx(Joint.R8)
        result.motor_cmd[lli].kp  = 100.0
        result.motor_cmd[lli].kd  = 1.0
        result.motor_cmd[lli].q   = wbclowcmd.left_gripper
        result.motor_cmd[lli].dq  = 0.0
        result.motor_cmd[lli].tau = 0.0
        result.motor_cmd[rli].kp  = 100.0
        result.motor_cmd[rli].kd  = 1.0
        result.motor_cmd[rli].q   = wbclowcmd.right_gripper
        result.motor_cmd[rli].dq  = 0.0
        result.motor_cmd[rli].tau = 0.0

        return result
