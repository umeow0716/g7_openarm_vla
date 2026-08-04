from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from g7_openarm_pinnzoo import PinnZooModel, inverse_dynamics, mass_matrix
from g7_openarm_utils import (
    ARM_MOTOR16_INDICES,
    ARM_POSITION_LOWER_RAD,
    ARM_POSITION_UPPER_RAD,
    ARM_VELOCITY_LIMIT_RAD_S,
    position_limited_velocity_bounds,
    quat_to_rotation_matrix,
)

from .config import config

if TYPE_CHECKING:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    from g7_openarm_idl import AMRCmd, Odom, OpenArmCmd


def odom_vdot_world_to_body(odom: Odom) -> npt.NDArray[np.float64]:
    quaternion = np.array(
        [
            odom.quaternion.w,
            odom.quaternion.x,
            odom.quaternion.y,
            odom.quaternion.z,
        ],
        dtype=np.float64,
    )
    rotation_world_to_body = quat_to_rotation_matrix(quaternion).T

    velocity_world = np.array(
        [odom.velocity.x, odom.velocity.y, odom.velocity.z],
        dtype=np.float64,
    )
    angular_velocity_world = np.array(
        [
            odom.angular_velocity.x,
            odom.angular_velocity.y,
            odom.angular_velocity.z,
        ],
        dtype=np.float64,
    )
    acceleration_world = np.array(
        [odom.vdot.x, odom.vdot.y, odom.vdot.z],
        dtype=np.float64,
    )
    angular_acceleration_world = np.array(
        [odom.angular_vdot.x, odom.angular_vdot.y, odom.angular_vdot.z],
        dtype=np.float64,
    )

    velocity_body = rotation_world_to_body @ velocity_world
    angular_velocity_body = rotation_world_to_body @ angular_velocity_world
    linear_vdot_body = rotation_world_to_body @ acceleration_world - np.cross(
        angular_velocity_body,
        velocity_body,
    )
    angular_vdot_body = rotation_world_to_body @ angular_acceleration_world

    return np.concatenate([linear_vdot_body, angular_vdot_body])


@dataclass(slots=True)
class ControllerConfig:
    wheel_radius_m: float = 0.052
    fl_pos: tuple[float, float] = (0.198, 0.13)
    fr_pos: tuple[float, float] = (0.198, -0.13)
    rl_pos: tuple[float, float] = (-0.198, 0.13)
    rr_pos: tuple[float, float] = (-0.198, -0.13)

    steer_hold_speed_m_s: float = 2e-2
    steer_branch_hysteresis_rad: float = np.deg2rad(15.0)
    steer_rate_limit_rad_s: float = np.deg2rad(240.0)
    steer_alignment_stop_rad: float = np.deg2rad(75.0)

    wheel_vel_limit_rad_s: float = 30.0

    base_idle_linear_threshold_m_s: float = 3e-2
    base_idle_angular_threshold_rad_s: float = 1e-2

    arm_torque_limit = np.array(
        [
            40.0,
            40.0,
            27.0,
            27.0,
            7.0,
            7.0,
            7.0,
            7.0,
            40.0,
            40.0,
            27.0,
            27.0,
            7.0,
            7.0,
            7.0,
            7.0,
        ],
        dtype=np.float64,
    )

    tau_static = np.array(
        [
            0.15,
            0.15,
            0.15,
            0.15,
            0.15,
            0.15,
            0.30,
            0.15,
        ]
        * 2,
        dtype=np.float64,
    )

    # --- PD / impedance control (arm) ---
    arm_pd_zeta: float = 1.2
    arm_pd_omega: float = 4.0

    # Extra multiplier on Kd only, applied AFTER the zeta*omega*sqrt(M)
    # formula. This is the knob to reach for on overshoot: it raises
    # damping without touching Kp (stiffness/bandwidth). Try raising
    # arm_pd_zeta toward ~1.0 first (same effect, more principled); reach
    # for this only if you need Kd beyond what zeta=1.0-1.5 gives you.
    arm_kd_boost: float = 1.0

    # Per-joint decoupling: applied ON TOP of the mass-matrix baseline
    # (Kp = ω²·M_diag, Kd = 2ζω·√M_diag), NOT instead of it -- this keeps
    # the automatic pose-dependent gain scheduling (an outstretched arm has
    # much higher effective shoulder inertia than a folded one) while still
    # letting you hand-correct individual joints whose real behaviour
    # (friction, backlash, cable drag) the diagonal mass-matrix model
    # doesn't capture. Index order matches motor_cmd[8:24]:
    #   [L1..L7, L_gripper, R1..R7, R_gripper]
    arm_kp_scale = np.ones(16, dtype=np.float64)
    arm_kd_scale = np.ones(16, dtype=np.float64)

    arm_pos_lead_torque_fraction: float = 0.5

    arm_kp_protocol_max: float = 500.0
    arm_kd_protocol_max: float = 6.0

    arm_kd_floor: float = 0.05


class Controller:
    def __init__(
        self,
        config: ControllerConfig | None = None,
        lib_path: str | None = None,
    ) -> None:
        self.config = config if config is not None else ControllerConfig()
        self.model = PinnZooModel(lib_path)

        self._arm_v_idx = [
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
        ]

        # Indices into the 16-length motor_cmd[8:24]-shaped arrays that are
        # grippers, not arm joints. PD gains/torque stay zero there; gripper
        # channel keeps whatever separate position handling cli.py already does.
        self._gripper_idx_16 = [7, 15]

        self._q_des_need_init = True
        self._q_des_16: npt.NDArray[np.float64] = np.zeros((16,), dtype=np.float64)

        # Stateful swerve branch selection. +1 means the direct-angle branch,
        # -1 means the angle+pi branch with reversed drive speed.
        self._prev_steer_target: npt.NDArray[np.float64] | None = None
        self._wheel_branch = np.ones(4, dtype=np.int8)

        self._wheel_xy = np.array(
            [
                self.config.fl_pos,
                self.config.fr_pos,
                self.config.rl_pos,
                self.config.rr_pos,
            ],
            dtype=np.float64,
        )

    def compute_arm_kd(
        self,
        M_diag: npt.NDArray[np.float64],
        zeta=0.7,
        omega=8.0,
    ):
        Kd = 2.0 * zeta * omega * np.sqrt(M_diag)
        return Kd[self._arm_v_idx]

    def compute_arm_kp(
        self,
        M_diag: npt.NDArray[np.float64],
        omega=8.0,
    ):
        Kp = (omega**2) * M_diag
        return Kp[self._arm_v_idx]

    @staticmethod
    def _to_motor16(v18: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Drop the two gripper-slot zero-paddings, same slicing as the
        existing tau_act -> tau_act_cmd conversion, to go from the 18-long
        generalized-velocity-ordered array to the 16-long motor_cmd[8:24]
        order."""
        return np.concatenate([v18[:8], v18[9:17]], dtype=np.float64)

    def update_base(
        self,
        lowstate: LowState_,
        amr_cmd: AMRCmd,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        base_command_is_idle = self.is_base_idle(amr_cmd)

        if base_command_is_idle:
            steer_pos_des = np.array(
                [motor.q for motor in lowstate.motor_state[:8:2]],
                dtype=np.float64,
            )
            wheel_vel_des = np.zeros((4,), dtype=np.float64)
            self._prev_steer_target = steer_pos_des.copy()
            return steer_pos_des, wheel_vel_des

        return self.swerve_inverse_kinematics(lowstate=lowstate, amr_cmd=amr_cmd)

    def update(self, lowstate: LowState_, odom: Odom, amr_cmd: AMRCmd, openarm_cmd: OpenArmCmd):
        x = PinnZooModel.build_x_lib(lowstate, odom)
        steer_pos_des, wheel_vel_des = self.update_base(lowstate, amr_cmd)

        dt = config.interval

        q_des, dq_des, Kp, Kd = self.compute_pd_targets(
            lowstate=lowstate,
            x=x,
            openarm_cmd=openarm_cmd,
            dt=dt,
        )
        tau_ff = self.compute_gravity_friction_ff(
            x=x,
            odom=odom,
            dq_des=dq_des,
        )

        return steer_pos_des, wheel_vel_des, q_des, dq_des, Kp, Kd, tau_ff

    def is_base_idle(self, amr_cmd: AMRCmd) -> bool:
        linear_speed = np.sqrt(amr_cmd.data[0] ** 2 + amr_cmd.data[1] ** 2)
        angular_speed = abs(amr_cmd.data[2])

        return (
            linear_speed < self.config.base_idle_linear_threshold_m_s
            and angular_speed < self.config.base_idle_angular_threshold_rad_s
        )

    def swerve_inverse_kinematics(
        self,
        lowstate: LowState_,
        amr_cmd: AMRCmd,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        vx, vy, wz = amr_cmd.data

        current_steer = np.array(
            [motor.q for motor in lowstate.motor_state[:8:2]],
            dtype=np.float64,
        )

        dt = config.interval

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
                candidate for candidate in candidates if candidate[2] == int(self._wheel_branch[i])
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
                    and keep_cost - best_cost < self.config.steer_branch_hysteresis_rad
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

            wheel_vel_des[i] = chosen_speed * alignment_scale / self.config.wheel_radius_m

        self._prev_steer_target = steer_pos_des.copy()

        max_abs_wheel_vel = float(np.max(np.abs(wheel_vel_des)))
        if max_abs_wheel_vel > self.config.wheel_vel_limit_rad_s:
            wheel_vel_des *= self.config.wheel_vel_limit_rad_s / max_abs_wheel_vel

        return steer_pos_des, wheel_vel_des

    def compute_pd_targets(
        self,
        lowstate: LowState_,
        x: npt.NDArray[np.float64],
        openarm_cmd: OpenArmCmd,
        dt: float,
    ) -> tuple[
        npt.NDArray[np.float64],  # q_des      (16,)
        npt.NDArray[np.float64],  # dq_des     (16,)
        npt.NDArray[np.float64],  # Kp         (16,)
        npt.NDArray[np.float64],  # Kd         (16,)
    ]:
        dq_des = np.array(
            openarm_cmd.data.copy(), dtype=np.float64
        )  # (16,), already in motor_cmd[8:24] order

        q_meas = np.array(
            [m.q for m in lowstate.motor_state[8:24]],
            dtype=np.float64,
        )

        if self._q_des_need_init:
            self._q_des_16 = q_meas.copy()
            self._q_des_need_init = False

        arm_q_meas = q_meas[ARM_MOTOR16_INDICES]
        arm_dq_lower, arm_dq_upper = position_limited_velocity_bounds(
            arm_q_meas,
            ARM_VELOCITY_LIMIT_RAD_S,
            dt,
        )
        dq_des[ARM_MOTOR16_INDICES] = np.clip(
            dq_des[ARM_MOTOR16_INDICES],
            arm_dq_lower,
            arm_dq_upper,
        )

        M = mass_matrix(self.model, x)
        M_diag = np.diag(M)
        Kp = self._to_motor16(self.compute_arm_kp(M_diag, omega=self.config.arm_pd_omega))
        Kd = self._to_motor16(
            self.compute_arm_kd(
                M_diag,
                zeta=self.config.arm_pd_zeta,
                omega=self.config.arm_pd_omega,
            )
        )
        Kp = Kp * self.config.arm_kp_scale
        Kd = Kd * self.config.arm_kd_scale * self.config.arm_kd_boost
        Kp = np.clip(Kp, 0.0, self.config.arm_kp_protocol_max)
        Kd = np.clip(Kd, 0.0, self.config.arm_kd_protocol_max)
        # Datasheet: Kd must not be 0 while Kp > 0, or the motor can
        # oscillate / lose control.
        Kd = np.where(Kp > 0.0, np.maximum(Kd, self.config.arm_kd_floor), Kd)

        # This is the anti-divergence mechanism. max_lead bounds how far
        # q_des is allowed to sit ahead of the MEASURED position, re-derived
        # every tick from the current (possibly clipped) Kp and the torque
        # budget you're willing to spend on the P-term alone. So no matter
        # how long the arm stalls, lags, or misses updates, the worst-case
        # P-torque is capped by design -- it can never run away.
        with np.errstate(divide="ignore", invalid="ignore"):
            max_lead = np.where(
                Kp > 1e-6,
                self.config.arm_pos_lead_torque_fraction
                * self.config.arm_torque_limit
                / np.maximum(Kp, 1e-6),
                0.0,
            )

        q_des_raw = self._q_des_16 + dq_des * dt
        q_des_raw[ARM_MOTOR16_INDICES] = np.clip(
            q_des_raw[ARM_MOTOR16_INDICES],
            ARM_POSITION_LOWER_RAD,
            ARM_POSITION_UPPER_RAD,
        )

        # Apply the anti-divergence lead bound after clipping the integrated
        # target. For a joint measured inside its valid range, this guarantees
        # q_des also remains inside the range. If feedback is already outside
        # the range, q_des moves inward gradually instead of causing a large
        # one-tick position error and P-torque spike.
        q_des = q_meas + np.clip(q_des_raw - q_meas, -max_lead, max_lead)

        for idx in self._gripper_idx_16:
            q_des[idx] = q_meas[idx]
            dq_des[idx] = 0.0
            Kp[idx] = 0.0
            Kd[idx] = 0.0

        self._q_des_16 = q_des.copy()

        return q_des, dq_des, Kp, Kd

    def compute_gravity_friction_ff(
        self,
        x: npt.NDArray[np.float64],
        odom: Odom,
        dq_des: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        vdot = np.concatenate(
            [
                odom_vdot_world_to_body(odom),  # (6,)
                [0.0] * 8,  # base wheels/steer, (8,)
                [0.0] * 18,  # arm accel = 0, (18,)
            ],
            dtype=np.float64,
        )

        tau_ff = inverse_dynamics(model=self.model, x=x, vdot=vdot)[6 + 8 :]

        if not np.all(np.isfinite(tau_ff)):
            self._q_des_16 = np.zeros((16,), dtype=np.float64)
            self._q_des_need_init = True
            raise RuntimeError(f"tau_ff has nan\n{tau_ff!r}")

        tau_ff_16 = self._to_motor16(tau_ff)

        want_move = np.abs(dq_des) > 4e-2
        tau_ff_16 = tau_ff_16 + self.config.tau_static * want_move * np.sign(dq_des)

        tau_ff_16 = np.clip(
            tau_ff_16,
            -self.config.arm_torque_limit,
            self.config.arm_torque_limit,
        )

        for idx in self._gripper_idx_16:
            tau_ff_16[idx] = 0.0

        return tau_ff_16
