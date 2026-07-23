from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import time
import platform
import numpy as np
import numpy.typing as npt

from typing import TYPE_CHECKING

from g7_openarm_pinnzoo import PinnZooModel, mass_matrix, inverse_dynamics

if TYPE_CHECKING:
    from g7_openarm_idl import AMRCmd, Odom, OpenArmCmd
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
MIN_CONTROLLER_DT_S = 1e-3


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

def odom_vdot_world_to_body(odom: "Odom") -> npt.NDArray[np.float64]:
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

    acceleration_world = np.array([
        odom.vdot.x,
        odom.vdot.y,
        odom.vdot.z,
    ], dtype=np.float64)

    angular_acceleration_world = np.array([
        odom.angular_vdot.x,
        odom.angular_vdot.y,
        odom.angular_vdot.z,
    ], dtype=np.float64)

    velocity_body = R_body_world @ velocity_world
    angular_velocity_body = R_body_world @ angular_velocity_world

    linear_vdot_body = (
        R_body_world @ acceleration_world
        - np.cross(angular_velocity_body, velocity_body)
    )

    angular_vdot_body = (
        R_body_world @ angular_acceleration_world
    )

    return np.concatenate([
        linear_vdot_body,
        angular_vdot_body,
    ])


@dataclass(slots=True)
class ControllerConfig:
    wheel_radius_m: float = 0.052
    fl_pos: tuple[float, float] = ( 0.198,  0.13)
    fr_pos: tuple[float, float] = ( 0.198, -0.13)
    rl_pos: tuple[float, float] = (-0.198,  0.13)
    rr_pos: tuple[float, float] = (-0.198, -0.13)

    min_wheel_speed_m_s: float = 1e-4
    steer_hold_speed_m_s: float = 2e-2
    steer_branch_hysteresis_rad: float = np.deg2rad(15.0)
    steer_rate_limit_rad_s: float = np.deg2rad(240.0)
    steer_alignment_stop_rad: float = np.deg2rad(75.0)

    wheel_vel_limit_rad_s: float = 30.0

    arm_acc_limit_rad_s2: float = 80.0
    
    base_steering_kp: float = 20.0
    base_steering_kd: float = 0.02
    base_wheel_kd: float = 5.0

    base_idle_linear_threshold_m_s: float = 3e-2
    base_idle_angular_threshold_rad_s: float = 1e-2
    
    arm_torque_limit = np.array([
        40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0, 7.0,
        40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0, 7.0,
    ], dtype=np.float64)
    
    tau_static = np.array([
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.30,
        0.15,
    ] * 2, dtype=np.float64)


class Controller:
    def __init__(
        self,
        config: ControllerConfig | None = None,
        lib_path: str | None = None,
    ) -> None:
        self.config = config if config is not None else ControllerConfig()
        self.lib_path = str(DEFAULT_LIB_PATH if lib_path is None else lib_path)
        self.model = PinnZooModel(self.lib_path)
        
        self._arm_act_idx = [
            0, 1,  2,  3,  4,  5,  6,
            8, 9, 10, 11, 12, 13, 14,
        ]
        self._arm_v_idx = [
            14, 15, 16, 17, 18, 19, 20, 21, 22,
            23, 24, 25, 26, 27, 28, 29, 30, 31,
        ]
        
        self._prev_arm_vel_des = np.zeros(18, dtype=np.float64)

        # Stateful swerve branch selection. +1 means the direct-angle branch,
        # -1 means the angle+pi branch with reversed drive speed.
        self._prev_steer_target: npt.NDArray[np.float64] | None = None
        self._wheel_branch = np.ones(4, dtype=np.int8)
        self._prev_swerve_time = time.perf_counter()

        self._wheel_xy = np.array(
            [
                self.config.fl_pos,
                self.config.fr_pos,
                self.config.rl_pos,
                self.config.rr_pos,
            ],
            dtype=np.float64,
        )
        
        self.prev_time = time.perf_counter()
    
    def compute_arm_kd(
        self,
        x: npt.NDArray[np.float64],
        zeta=0.7,
        omega=8.0,
    ):
        M = mass_matrix(self.model, x)
        M_diag = np.diag(M)
        Kd = 2.0 * zeta * omega * np.sqrt(M_diag)
        return Kd[self._arm_v_idx]
    
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

    def update(
        self,
        lowstate: "LowState_",
        odom: "Odom",
        amr_cmd: "AMRCmd",
        openarm_cmd: "OpenArmCmd"
    ):
        x = self.build_x_lib(lowstate, odom)
        base_command_is_idle = self.is_base_idle(amr_cmd)
        
        if base_command_is_idle:
            steer_pos_des = x[7:7+8:2].copy()
            wheel_vel_des = np.zeros((4,), dtype=np.float64)
            # Synchronize the remembered target to the stopped physical pose.
            self._prev_steer_target = steer_pos_des.copy()
        else:
            steer_pos_des, wheel_vel_des = self.swerve_inverse_kinematics(
                lowstate=lowstate,
                amr_cmd=amr_cmd
            )

        acc_act_des = self.desired_actuator_acceleration(
            x=x,
            openarm_cmd=openarm_cmd,
        )
        
        vdot = np.concatenate([
            odom_vdot_world_to_body(odom), # (6,)
            [ 0.0 ] * 8,  # (8,)
            acc_act_des,  # (18,)
        ], dtype=np.float64)
        tau_act = inverse_dynamics(
            model=self.model,
            x=x,
            vdot=vdot,
        )[6+8:]
        tau_act_cmd = np.concatenate([
            tau_act[:8],
            tau_act[9:17],
        ], dtype=np.float64) # (16,)
        
        if not np.all(np.isfinite(tau_act)):
            self._prev_arm_vel_des[:] = 0.0
            raise RuntimeError(f"tau_act has nan\n{repr(tau_act)}")
        
        want_move = np.abs(openarm_cmd.data) > 4e-2
        tau_bias = self.config.tau_static * want_move * np.sign(openarm_cmd.data)
        tau_act_cmd += tau_bias
        tau_act_cmd[7]  = 0.0
        tau_act_cmd[15] = 0.0
        
        return steer_pos_des, wheel_vel_des, tau_act_cmd

    def is_base_idle(
        self,
        amr_cmd: 'AMRCmd'
    ) -> bool:
        linear_speed  = np.sqrt(amr_cmd.data[0] ** 2 + amr_cmd.data[1] ** 2)
        angular_speed = abs(amr_cmd.data[2])
        
        return (
            linear_speed < self.config.base_idle_linear_threshold_m_s
            and angular_speed < self.config.base_idle_angular_threshold_rad_s
        )

    def swerve_inverse_kinematics(
        self,
        lowstate: 'LowState_',
        amr_cmd: 'AMRCmd',
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        vx, vy, wz = amr_cmd.data

        current_steer = np.array(
            [motor.q for motor in lowstate.motor_state[:8:2]],
            dtype=np.float64,
        )

        now = time.perf_counter()
        dt = max(now - self._prev_swerve_time, MIN_CONTROLLER_DT_S)
        self._prev_swerve_time = now

        if self._prev_steer_target is None:
            self._prev_steer_target = current_steer.copy()

        prev_target = self._prev_steer_target
        steer_pos_des = prev_target.copy()
        wheel_vel_des = np.zeros(4, dtype=np.float64)

        steer_limit = np.deg2rad(100.0)
        max_steer_step = self.config.steer_rate_limit_rad_s * dt

        for i, (wheel_x, wheel_y) in enumerate(self._wheel_xy):
            wheel_vx = vx - wz * wheel_y
            wheel_vy = vy + wz * wheel_x
            speed = float(np.hypot(wheel_vx, wheel_vy))

            if speed < self.config.steer_hold_speed_m_s:
                steer_pos_des[i] = prev_target[i]
                wheel_vel_des[i] = 0.0
                continue

            base_angle = float(np.atan2(wheel_vy, wheel_vx))

            # (angle, signed linear speed, branch)
            candidates: list[tuple[float, float, int]] = []
            for k in range(-2, 3):
                candidate_angle = base_angle + k * np.pi
                if -steer_limit <= candidate_angle <= steer_limit:
                    branch = 1 if k % 2 == 0 else -1
                    candidates.append((candidate_angle, branch * speed, branch))

            if not candidates:
                steer_pos_des[i] = prev_target[i]
                wheel_vel_des[i] = 0.0
                continue

            best = min(
                candidates,
                key=lambda candidate: abs(candidate[0] - current_steer[i]),
            )

            previous_branch_candidates = [
                candidate for candidate in candidates
                if candidate[2] == int(self._wheel_branch[i])
            ]

            if previous_branch_candidates:
                keep = min(
                    previous_branch_candidates,
                    key=lambda candidate: abs(candidate[0] - current_steer[i]),
                )
                keep_cost = abs(keep[0] - current_steer[i])
                best_cost = abs(best[0] - current_steer[i])

                if (
                    best[2] != int(self._wheel_branch[i])
                    and keep_cost - best_cost
                    < self.config.steer_branch_hysteresis_rad
                ):
                    chosen = keep
                else:
                    chosen = best
            else:
                chosen = best

            chosen_angle, chosen_speed, chosen_branch = chosen
            self._wheel_branch[i] = chosen_branch

            target_delta = chosen_angle - prev_target[i]
            steer_pos_des[i] = prev_target[i] + np.clip(
                target_delta,
                -max_steer_step,
                max_steer_step,
            )

            steer_error = chosen_angle - current_steer[i]
            if abs(steer_error) >= self.config.steer_alignment_stop_rad:
                alignment_scale = 0.0
            else:
                alignment_scale = max(0.0, float(np.cos(steer_error)))

            wheel_vel_des[i] = (
                chosen_speed
                * alignment_scale
                / self.config.wheel_radius_m
            )

        self._prev_steer_target = steer_pos_des.copy()

        max_abs_wheel_vel = float(np.max(np.abs(wheel_vel_des)))
        if max_abs_wheel_vel > self.config.wheel_vel_limit_rad_s:
            wheel_vel_des *= (
                self.config.wheel_vel_limit_rad_s
                / max_abs_wheel_vel
            )

        return steer_pos_des, wheel_vel_des

    def desired_actuator_acceleration(
        self,
        x: npt.NDArray[np.float64],
        openarm_cmd: 'OpenArmCmd'
    ) -> npt.NDArray[np.float64]:
        now = time.perf_counter()
        dt = max(now - self.prev_time, MIN_CONTROLLER_DT_S)
        
        arm_vel_des = np.concatenate([
            openarm_cmd.data[:8],
            [ 0.0 ],
            openarm_cmd.data[8:],
            [ 0.0 ],
        ], dtype=np.float64)
        arm_acc_ff = (arm_vel_des - self._prev_arm_vel_des) / dt
        
        self._prev_arm_vel_des[:] = arm_vel_des
        self.prev_time = now
        
        arm_vel_err = arm_vel_des - x[-18:]
        
        KDs = self.compute_arm_kd(
            x,
            zeta=0.7,
            omega=8.0,
        )
        
        acc = np.zeros(18, dtype=np.float64)
        acc = arm_acc_ff + KDs * arm_vel_err
        acc[7:9] = 0.0
        acc[16:] = 0.0
        acc = np.clip(
            acc,
            -self.config.arm_acc_limit_rad_s2,
            self.config.arm_acc_limit_rad_s2,
        )
        return acc

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi