from __future__ import annotations

import numpy as np

from g7_openarm_utils.layout.control_layout import (
    Joint,
    low_idx,
)
from unitree_sdk2py.idl.default import HGLowState_

class WheelKinematic:
    def __init__(self):
        self._steer_joint = [Joint.AMR_FL, Joint.AMR_FR, Joint.AMR_RL, Joint.AMR_RR]
        
        self._wheel_radius_m = 0.052
        self._wheel_xy = np.array(
            [
                ( 0.198,  0.13),
                ( 0.198, -0.13),
                (-0.198,  0.13),
                (-0.198, -0.13),
            ],
            dtype=np.float64,
        )
        self._base_idle_linear_threshold_m_s: float = 3e-2
        self._base_idle_angular_threshold_rad_s: float = 1e-2

    def choose_best_pose(
        self,
        lowstate: "HGLowState_",
        vx: float,
        vy: float,
        wz: float,
    ) -> dict[Joint, float]:
        if self.is_idle(vx, vy, wz):
            return {
                Joint.AMR_FL: lowstate.motor_state[low_idx(Joint.AMR_FL)].q,
                Joint.AMR_FLW: 0.0,
                Joint.AMR_FR: lowstate.motor_state[low_idx(Joint.AMR_FR)].q,
                Joint.AMR_FRW: 0.0,
                Joint.AMR_RL: lowstate.motor_state[low_idx(Joint.AMR_RL)].q,
                Joint.AMR_RLW: 0.0,
                Joint.AMR_RR: lowstate.motor_state[low_idx(Joint.AMR_RR)].q,
                Joint.AMR_RRW: 0.0,
            }

        steer_pos_des, wheel_vel_des = self.swerve_inverse_kinematics(
            lowstate,
            vx,
            vy,
            wz,
        )
        
        return {
            Joint.AMR_FL: steer_pos_des[0],
            Joint.AMR_FLW: wheel_vel_des[0],
            Joint.AMR_FR: steer_pos_des[1],
            Joint.AMR_FRW: wheel_vel_des[1],
            Joint.AMR_RL: steer_pos_des[2],
            Joint.AMR_RLW: wheel_vel_des[2],
            Joint.AMR_RR: steer_pos_des[3],
            Joint.AMR_RRW: wheel_vel_des[3]
        }

    def swerve_inverse_kinematics(
        self,
        lowstate: "HGLowState_",
        vx: float,
        vy: float,
        wz: float,
    ):
        current_steer = np.zeros((4,), dtype=np.float64)
        for i, joint in enumerate(self._steer_joint):
            current_steer[i] = lowstate.motor_state[low_idx(joint)].q 

        steer_limit = np.deg2rad(90.0)

        steer_pos_des = current_steer.copy()
        wheel_vel_des = np.zeros((4,), dtype=np.float64)
        for i, (wheel_x, wheel_y) in enumerate(self._wheel_xy):
            wheel_vx = vx - wz * wheel_y
            wheel_vy = vy + wz * wheel_x

            speed = float(np.hypot(wheel_vx, wheel_vy))
            base_angle = float(np.atan2(wheel_vy, wheel_vx))

            candidates: list[tuple[float, float]] = []

            for k in (-1, 0, 1):
                candidate_angle = base_angle + k * np.pi

                if -steer_limit <= candidate_angle <= steer_limit:
                    wheel_speed = speed if k % 2 == 0 else -speed
                    candidates.append((candidate_angle, wheel_speed))


            best_angle, best_speed = min(
                candidates,
                key=lambda candidate: abs(candidate[0] - current_steer[i]),
            )

            steer_pos_des[i] = best_angle
            wheel_vel_des[i] = best_speed / self._wheel_radius_m

        return steer_pos_des, wheel_vel_des
    
    def is_idle(self, vx: float, vy: float, wz: float) -> bool:
        linear_speed = np.hypot(vx, vy)
        angular_speed = abs(wz)

        return (
            linear_speed < self._base_idle_linear_threshold_m_s
            and angular_speed < self._base_idle_angular_threshold_rad_s
        )
    