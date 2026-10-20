from __future__ import annotations

import time
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

from .config import config

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
        self._last_debug_t = 0.0

    def initialize(self, lowstate: "HGLowState_"):
        self.reference_generator.initialize(lowstate)

    
    def compute_arm_kd(
        self,
        M_diag: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        omega = np.ones_like(M_diag, dtype=np.float64)
        zeta = np.ones_like(M_diag, dtype=np.float64)

        for joint in ARM_JOINT:
            idx = pinnzoo_vdot_idx(joint)
            omega[idx] = 2.0 * np.pi * config.natural_frequency[joint]
            zeta[idx] = config.damping_ratio[joint]

        Kd = 2.0 * zeta * omega * M_diag

        return {
            joint: float(Kd[pinnzoo_vdot_idx(joint)])
            for joint in ARM_JOINT
        }

    def compute_arm_kp(
        self,
        M_diag: npt.NDArray[np.float64],
    ):
        omega = np.ones_like(M_diag, dtype=np.float64)

        for joint in ARM_JOINT:
            idx = pinnzoo_vdot_idx(joint)
            omega[idx] = 2.0 * np.pi * config.natural_frequency[joint]

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

        vdot = np.zeros((self.model.nv,), dtype=np.float64)
        vdot[:6] = (
            odom.vdot.x, odom.vdot.y, odom.vdot.z,
            odom.angular_vdot.x, odom.angular_vdot.y, odom.angular_vdot.z,
        )

        arm_idx = [pinnzoo_vdot_idx(joint) for joint in ARM_JOINT]
        tau_bias = inverse_dynamics(self.model, lib_x, vdot)[arm_idx]
        dq_des = np.array(wbclowcmd.dq_des, dtype=np.float64)
        ref = self.reference_generator.generate(
            lowstate, dq_des, M[np.ix_(arm_idx, arm_idx)], tau_bias,
        )

        for joint in ARM_JOINT:
            vdot[pinnzoo_vdot_idx(joint)] = ref.ddq[joint]
        tau_ff = inverse_dynamics(self.model, lib_x, vdot)
        a_fb = np.zeros((self.model.nv,), dtype=np.float64)
        
        for joint in ARM_JOINT:
            state = lowstate.motor_state[low_idx(joint)]
            j = pinnzoo_vdot_idx(joint)

            omega = 2.0 * np.pi * config.natural_frequency[joint]
            zeta = config.damping_ratio[joint]

            a_fb[j] = \
                omega**2 * (ref.q[joint] - state.q) + \
                2.0 * zeta * omega * (ref.dq[joint] - state.dq)

        tau_cpl = M @ a_fb - M_diag * a_fb

        tau_raw = np.zeros((14,), dtype=np.float64)
        for i, joint in enumerate(ARM_JOINT):
            low_motor_state = lowstate.motor_state[low_idx(joint)]
            q_mea_i  = low_motor_state.q
            dq_mea_i = low_motor_state.dq
            tau_raw[i] = \
                Kp[joint] * (ref.q[joint] - q_mea_i) + \
                Kd[joint] * (ref.dq[joint] - dq_mea_i) + \
                tau_ff[pinnzoo_vdot_idx(joint)] + \
                tau_cpl[pinnzoo_vdot_idx(joint)]


        # now = time.monotonic()
        
        # if now - self._last_debug_t >= 0.05:
        #     self._last_debug_t = now

        #     for name, joint, i in (
        #         ("L7", Joint.L7, 6),
        #         ("R7", Joint.R7, 13),
        #     ):
        #         state = lowstate.motor_state[low_idx(joint)]

        #         print(
        #             f"t={now:.3f} {name} "
        #             f"q_ref={ref.q[joint]:+.5f} "
        #             f"q_meas={state.q:+.5f} "
        #             f"dq_ref={ref.dq[joint]:+.5f} "
        #             f"dq_meas={state.dq:+.5f} "
        #             f"kp={Kp[joint]:.2f} "
        #             f"kd={Kd[joint]:.2f} "
        #             f"tau_ff={tau_ff[pinnzoo_vdot_idx(joint)]:+.3f} "
        #             f"tau_cpl={tau_cpl[pinnzoo_vdot_idx(joint)]:+.3f} "
        #             f"tau_cmd={tau_raw[i]:+.3f} "
        #             f"tau_est={getattr(state, 'tau_est', None)}"
        #         )

        result = HGLowCmd_default()
        
        wheel_enabled = True
        for joint in BASE_JOINT:
            idx = low_idx(joint)
            if Joint.is_wheel(joint):
               continue
            if abs(best_wheel[joint] - lowstate.motor_state[idx].q) > np.deg2rad(5):
                wheel_enabled = False

        for joint in BASE_JOINT:
            idx = low_idx(joint)
            if Joint.is_wheel(joint):
                result.motor_cmd[idx].kp  = 0.0
                result.motor_cmd[idx].kd  = 5.0
                result.motor_cmd[idx].q   = 0.0
                result.motor_cmd[idx].dq  = best_wheel[joint] if wheel_enabled else 0.0
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
            result.motor_cmd[idx].tau = tau_ff[pinnzoo_vdot_idx(joint)] + tau_cpl[pinnzoo_vdot_idx(joint)]

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
