from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


FloatArray = npt.NDArray[np.float64]


def _rotation_body_to_world(q_wxyz: FloatArray) -> FloatArray:
    q = np.asarray(q_wxyz, dtype=np.float64)

    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("IMU quaternion norm is zero")

    q = q / norm
    w, x, y, z = q

    return np.array([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class AMREKFConfig:
    wheel_velocity_std: float = 0.05
    wheel_wz_std: float = np.deg2rad(3.0)
    gyro_wz_std: float = np.deg2rad(0.5)

    linear_velocity_process_std: float = 0.5
    angular_velocity_process_std: float = np.deg2rad(10.0)
    gyro_bias_walk_std: float = np.deg2rad(0.03)


@dataclass(frozen=True, slots=True)
class WorldState:
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
        config: AMREKFConfig | None = None,
    ) -> None:
        self.config = config if config is not None else AMREKFConfig()
        self.wheel_radius = wheel_radius

        # FL, FR, RL, RR
        self.wheel_position = np.array([
            [front_x, left_y],
            [front_x, right_y],
            [rear_x, left_y],
            [rear_x, right_y],
        ], dtype=np.float64)

        # [px_world, py_world, vx_world, vy_world, wz_body, gyro_z_bias]
        self.x = np.zeros(6, dtype=np.float64)

        self.P = np.diag([
            0.01,
            0.01,
            0.05,
            0.05,
            np.deg2rad(1.0),
            np.deg2rad(1.0),
        ]) ** 2

        self.previous_linear_velocity_body: FloatArray | None = None
        self.previous_wz_body: float | None = None

    def update(self, lowstate: LowState_, dt: float) -> WorldState:
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        dt = min(dt, 0.1)

        quat = np.asarray(
            lowstate.imu_state.quaternion,
            dtype=np.float64,
        ).copy()

        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-12:
            raise ValueError("IMU quaternion norm is zero")
        quat /= quat_norm

        rotation = _rotation_body_to_world(quat)

        wheel_velocity_body = self._swerve_velocity(lowstate)

        wheel_linear_velocity_world = rotation @ np.array([
            wheel_velocity_body[0],
            wheel_velocity_body[1],
            0.0,
        ], dtype=np.float64)

        gyro_wz_body = float(lowstate.imu_state.gyroscope[2])

        # Constant-velocity prediction.
        self.x[0] += self.x[2] * dt
        self.x[1] += self.x[3] * dt

        F = np.eye(6, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt

        Q = np.diag([
            (0.02 * dt) ** 2,
            (0.02 * dt) ** 2,
            (self.config.linear_velocity_process_std * dt) ** 2,
            (self.config.linear_velocity_process_std * dt) ** 2,
            (self.config.angular_velocity_process_std * dt) ** 2,
            self.config.gyro_bias_walk_std ** 2 * dt,
        ])

        self.P = F @ self.P @ F.T + Q

        # Wheel wz measures true yaw rate.
        # Gyroscope wz measures true yaw rate plus gyro bias.
        measurement = np.array([
            wheel_linear_velocity_world[0],
            wheel_linear_velocity_world[1],
            wheel_velocity_body[2],
            gyro_wz_body,
        ], dtype=np.float64)

        prediction = np.array([
            self.x[2],
            self.x[3],
            self.x[4],
            self.x[4] + self.x[5],
        ], dtype=np.float64)

        H = np.zeros((4, 6), dtype=np.float64)
        H[0, 2] = 1.0
        H[1, 3] = 1.0
        H[2, 4] = 1.0
        H[3, 4] = 1.0
        H[3, 5] = 1.0

        R = np.diag([
            self.config.wheel_velocity_std ** 2,
            self.config.wheel_velocity_std ** 2,
            self.config.wheel_wz_std ** 2,
            self.config.gyro_wz_std ** 2,
        ])

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

        linear_velocity_world = np.array([
            self.x[2],
            self.x[3],
            0.0,
        ], dtype=np.float64)

        linear_velocity_body = rotation.T @ linear_velocity_world

        wz_body = float(self.x[4])
        angular_velocity_body = np.array([
            0.0,
            0.0,
            wz_body,
        ], dtype=np.float64)

        if self.previous_linear_velocity_body is None:
            linear_vdot_body = np.zeros(3, dtype=np.float64)
            angular_vdot_body = np.zeros(3, dtype=np.float64)
        else:
            linear_vdot_body = (
                linear_velocity_body
                - self.previous_linear_velocity_body
            ) / dt

            angular_vdot_body = np.array([
                0.0,
                0.0,
                (wz_body - self.previous_wz_body) / dt,
            ], dtype=np.float64)

        self.previous_linear_velocity_body = linear_velocity_body.copy()
        self.previous_wz_body = wz_body

        angular_velocity_world = rotation @ angular_velocity_body

        # Convert derivative of body-frame velocity components into classical
        # acceleration of the base origin, expressed in world frame.
        linear_acceleration_world = rotation @ (
            linear_vdot_body
            + np.cross(
                angular_velocity_body,
                linear_velocity_body,
            )
        )

        angular_acceleration_world = rotation @ angular_vdot_body

        return WorldState(
            x=float(self.x[0]),
            y=float(self.x[1]),
            z=0.0,
            quat=quat,
            vx=float(linear_velocity_world[0]),
            vy=float(linear_velocity_world[1]),
            vz=0.0,
            angular_velocity=angular_velocity_world,
            vdot=linear_acceleration_world,
            angular_vdot=angular_acceleration_world,
        )

    def _swerve_velocity(self, lowstate: LowState_) -> FloatArray:
        steering = np.array([
            motor.q for motor in lowstate.motor_state[:8:2]
        ], dtype=np.float64)

        wheel_speed = np.array([
            motor.dq * self.wheel_radius
            for motor in lowstate.motor_state[1:8:2]
        ], dtype=np.float64)

        A = np.zeros((8, 3), dtype=np.float64)
        b = np.zeros(8, dtype=np.float64)

        for i, ((x, y), angle, speed) in enumerate(
            zip(
                self.wheel_position,
                steering,
                wheel_speed,
            )
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
