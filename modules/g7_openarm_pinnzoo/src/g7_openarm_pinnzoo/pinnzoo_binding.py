import os
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from cffi import FFI

from .pinnzoo_utils import get_arch

if TYPE_CHECKING:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    from g7_openarm_idl import Odom


def odom_velocity_world_to_body(
    odom: "Odom",
) -> npt.NDArray[np.float64]:
    qw = odom.quaternion.w
    qx = odom.quaternion.x
    qy = odom.quaternion.y
    qz = odom.quaternion.z

    R_world_body = np.array(
        [
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
        ],
        dtype=np.float64,
    )

    R_body_world = R_world_body.T

    velocity_world = np.array(
        [
            odom.velocity.x,
            odom.velocity.y,
            odom.velocity.z,
        ],
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

    velocity_body = R_body_world @ velocity_world
    angular_velocity_body = R_body_world @ angular_velocity_world

    return np.concatenate(
        [
            velocity_body,
            angular_velocity_body,
        ]
    )


class PinnZooModel:
    def __init__(self, lib_path: str):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"file `{lib_path}` not found!")

        self.lib_path = lib_path
        self.ffi = FFI()

        self.ffi.cdef("""
extern const char* config_names[];
extern const char* vel_names[];
extern const char* torque_names[];
extern const char* kinematics_bodies[];
void M_func_wrapper(double* x_in, double* M_out);
void kinematics_wrapper(double* x, double* locs);
void kinematics_jacobian_wrapper(double* x, double* J);
void forward_dynamics_wrapper(double* x_in, double* tau_in, double* vdot_out);
void forward_dynamics_deriv_wrapper(double* x_in, double* tau_in, double* dvdot_dx_out, double* dvdout_dtau_out);
void inverse_dynamics_wrapper(double* x_in, double* vdot_in, double* tau_out);
void dynamics_deriv_wrapper(double* x_in, double* tau_in, double* dxdot_dx_out, double* dxdout_dtau_out);
        """)

        self.lib = self.ffi.dlopen(os.path.abspath(self.lib_path))

        self.nq = self._get_c_array_len(self.lib.config_names)  # type: ignore
        self.nv = self._get_c_array_len(self.lib.vel_names)  # type: ignore
        self.nx = self.nq + self.nv
        self.nu = self.nv

        self.bodies_count = self._get_c_array_len(self.lib.kinematics_bodies)  # type: ignore

    @cached_property
    def kinematics_size(self):
        if "quat" in self.lib_path:
            return 7 * self.bodies_count
        else:
            return 3 * self.bodies_count

    def _get_c_array_len(self, ptr):
        count = 0
        while ptr[count] != self.ffi.NULL:
            count += 1
        return count

    @staticmethod
    def get_default_lib_path():
        return (
            Path(__file__).resolve().parent.parent.parent.parent.parent
            / "include"
            / f"libg7_openarm_quat_{get_arch()}.so"
        )

    @staticmethod
    def build_x_lib(lowstate: "LowState_", odom: "Odom") -> npt.NDArray[np.float64]:
        motor_state = lowstate.motor_state

        position = np.array([odom.position.x, odom.position.y, odom.position.z], dtype=np.float64)
        quat = np.array(
            [odom.quaternion.w, odom.quaternion.x, odom.quaternion.y, odom.quaternion.z],
            dtype=np.float64,
        )

        q_0_14 = np.array([m.q for m in motor_state[0:15]], dtype=np.float64)
        q_16_22 = np.array([m.q for m in motor_state[16:23]], dtype=np.float64)

        dq_0_14 = np.array([m.dq for m in motor_state[0:15]], dtype=np.float64)
        dq_16_22 = np.array([m.dq for m in motor_state[16:23]], dtype=np.float64)

        return np.concatenate(
            (
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
            )
        )
