from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from g7_openarm_utils import (
    quat_normalize,
    quat_to_rotation_matrix,
)
from g7_openarm_utils.layout import Joint, low_idx
from unitree_sdk2py.idl.default import HGIMUState_, HGLowState_

from .config import config

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AMREKFConfig:
    wheel_velocity_std: float = 0.05
    wheel_wz_std: float = np.deg2rad(3.0)
    gyro_wz_std: float = np.deg2rad(0.5)

    linear_velocity_process_std: float = 0.5
    angular_velocity_process_std: float = np.deg2rad(10.0)
    gyro_bias_walk_std: float = np.deg2rad(0.03)


@dataclass(frozen=True, slots=True)
class State:
    x: float
    y: float
    z: float

    quat: FloatArray

    vx: float
    vy: float
    vz: float
    angular_velocity: FloatArray

    vdot: FloatArray
    angular_vdot: FloatArray


class AMREKF:
    """
    Planar swerve-drive EKF.

    Inputs:
      - Wheel odometry: vx_body, vy_body, wz_body
      - IMU quaternion: [w, x, y, z], body -> world
      - IMU gyroscope Z: wz_body + bias

    Accelerometer is intentionally not used.

    State:
      [px_world, py_world, vx_world, vy_world, wz_body, gyro_z_bias]
    """

    def __init__(
        self,
        wheel_radius: float = 0.052,
        front_x: float = 0.198,
        rear_x: float = -0.198,
        left_y: float = 0.13,
        right_y: float = -0.13,
    ) -> None:
        self.config = AMREKFConfig()
        self.wheel_radius = wheel_radius
        self.steer_joint = [ Joint.AMR_FL, Joint.AMR_FR, Joint.AMR_RL, Joint.AMR_RR ]
        self.wheel_joint = [ Joint.AMR_FLW, Joint.AMR_FRW, Joint.AMR_RLW, Joint.AMR_RRW]
        self.delta_t = 1.0 / config.hz

        # FL, FR, RL, RR
        self.wheel_position = np.array(
            [
                [front_x, left_y],
                [front_x, right_y],
                [rear_x, left_y],
                [rear_x, right_y],
            ],
            dtype=np.float64,
        )

        # [px_world, py_world, vx_world, vy_world, wz_body, gyro_z_bias]
        self.x = np.zeros(6, dtype=np.float64)

        self.P = (
            np.diag(
                [
                    0.01,
                    0.01,
                    0.05,
                    0.05,
                    np.deg2rad(1.0),
                    np.deg2rad(1.0),
                ]
            )
            ** 2
        )

        self.previous_linear_velocity_body: npt.NDArray[np.float64] | None = None
        self.previous_angular_velocity_body: npt.NDArray[np.float64] | None = None

    def update(
        self,
        imustate: HGIMUState_,
        lowstate: HGLowState_,
    ) -> State:
        quat = np.asarray(
            imustate.quaternion,
            dtype=np.float64,
        ).copy()

        quat = quat_normalize(quat)
        rotation = quat_to_rotation_matrix(quat)

        wheel_velocity_body = self._swerve_velocity(lowstate)

        wheel_linear_velocity_world = rotation @ np.array(
            [
                wheel_velocity_body[0],
                wheel_velocity_body[1],
                0.0,
            ],
            dtype=np.float64,
        )

        gyro_wz_body = float(imustate.gyroscope[2])

        # Constant-velocity prediction.
        self.x[0] += self.x[2] * self.delta_t
        self.x[1] += self.x[3] * self.delta_t

        F = np.eye(6, dtype=np.float64)
        F[0, 2] = self.delta_t
        F[1, 3] = self.delta_t

        Q = np.diag(
            [
                (0.02 * self.delta_t) ** 2,
                (0.02 * self.delta_t) ** 2,
                (self.config.linear_velocity_process_std * self.delta_t) ** 2,
                (self.config.linear_velocity_process_std * self.delta_t) ** 2,
                (self.config.angular_velocity_process_std * self.delta_t) ** 2,
                self.config.gyro_bias_walk_std**2 * self.delta_t,
            ]
        )

        self.P = F @ self.P @ F.T + Q

        # Wheel wz measures true yaw rate.
        # Gyroscope wz measures true yaw rate plus gyro bias.
        measurement = np.array(
            [
                wheel_linear_velocity_world[0],
                wheel_linear_velocity_world[1],
                wheel_velocity_body[2],
                gyro_wz_body,
            ],
            dtype=np.float64,
        )

        prediction = np.array(
            [
                self.x[2],
                self.x[3],
                self.x[4],
                self.x[4] + self.x[5],
            ],
            dtype=np.float64,
        )

        H = np.zeros((4, 6), dtype=np.float64)
        H[0, 2] = 1.0
        H[1, 3] = 1.0
        H[2, 4] = 1.0
        H[3, 4] = 1.0
        H[3, 5] = 1.0

        R = np.diag(
            [
                self.config.wheel_velocity_std**2,
                self.config.wheel_velocity_std**2,
                self.config.wheel_wz_std**2,
                self.config.gyro_wz_std**2,
            ]
        )

        innovation = measurement - prediction
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(
            S,
            H @ self.P,
        ).T

        self.x += K @ innovation

        I = np.eye(6, dtype=np.float64)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        linear_velocity_world_xy = np.array(
            [
                self.x[2],
                self.x[3],
            ],
            dtype=np.float64,
        )

        rotation_xy = rotation[:2, :2]

        linear_velocity_body_xy = np.linalg.solve(
            rotation_xy,
            linear_velocity_world_xy,
        )

        linear_velocity_body = np.array(
            [
                linear_velocity_body_xy[0],
                linear_velocity_body_xy[1],
                0.0,
            ],
            dtype=np.float64,
        )

        wz_body = float(self.x[4])
        angular_velocity_body = np.asarray(
            imustate.gyroscope,
            dtype=np.float64,
        ).copy()

        angular_velocity_body[2] = wz_body

        if self.previous_linear_velocity_body is None or \
            self.previous_angular_velocity_body is None:
            linear_vdot_body = np.zeros(3, dtype=np.float64)
            angular_vdot_body = np.zeros(3, dtype=np.float64)
        else:
            linear_vdot_body = (linear_velocity_body - self.previous_linear_velocity_body) / self.delta_t
            angular_vdot_body = (angular_velocity_body - self.previous_angular_velocity_body) / self.delta_t

        self.previous_linear_velocity_body = linear_velocity_body.copy()
        self.previous_angular_velocity_body = angular_velocity_body.copy()

        return State(
            x=float(self.x[0]),
            y=float(self.x[1]),
            z=0.0,
            quat=quat,
            vx=float(linear_velocity_body[0]),
            vy=float(linear_velocity_body[1]),
            vz=0.0,
            angular_velocity=angular_velocity_body,
            vdot=linear_vdot_body,
            angular_vdot=angular_vdot_body,
        )

    def _swerve_velocity(self, lowstate: HGLowState_) -> FloatArray:
        steering = [lowstate.motor_state[low_idx(joint)].q for joint in self.steer_joint]
        wheel_speed = [lowstate.motor_state[low_idx(joint)].dq * self.wheel_radius for joint in self.wheel_joint]

        A = np.zeros((8, 3), dtype=np.float64)
        b = np.zeros(8, dtype=np.float64)

        for i, ((x, y), angle, speed) in enumerate(
            zip(self.wheel_position, steering, wheel_speed, strict=True)
        ):
            c = np.cos(angle)
            s = np.sin(angle)

            A[2 * i] = [
                c,
                s,
                -y * c + x * s,
            ]
            b[2 * i] = speed

            A[2 * i + 1] = [
                -s,
                c,
                y * s + x * c,
            ]

        return np.linalg.lstsq(A, b, rcond=None)[0]
