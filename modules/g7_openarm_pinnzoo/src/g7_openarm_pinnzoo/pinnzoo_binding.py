from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from cffi import FFI

from g7_openarm_utils import quat_to_rotation_matrix

from .resources import native_library_path

if TYPE_CHECKING:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    from g7_openarm_idl import Odom


def odom_velocity_world_to_body(
    odom: Odom,
) -> npt.NDArray[np.float64]:
    quaternion = np.array(
        [
            odom.quaternion.w,
            odom.quaternion.x,
            odom.quaternion.y,
            odom.quaternion.z,
        ],
        dtype=np.float64,
    )
    rotation_body_to_world = quat_to_rotation_matrix(quaternion)

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

    rotation_world_to_body = rotation_body_to_world.T
    return np.concatenate(
        [
            rotation_world_to_body @ velocity_world,
            rotation_world_to_body @ angular_velocity_world,
        ]
    )


class PinnZooModel:
    def __init__(self, lib_path: str | Path | None = None) -> None:
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
void forward_dynamics_deriv_wrapper(
    double* x_in,
    double* tau_in,
    double* dvdot_dx_out,
    double* dvdout_dtau_out
);
void inverse_dynamics_wrapper(double* x_in, double* vdot_in, double* tau_out);
void dynamics_deriv_wrapper(
    double* x_in,
    double* tau_in,
    double* dxdot_dx_out,
    double* dxdout_dtau_out
);
        """)

        if lib_path is None:
            with native_library_path() as packaged_path:
                self.lib_path = str(packaged_path)
                self.lib = self.ffi.dlopen(self.lib_path)
        else:
            resolved_path = Path(lib_path).expanduser().resolve()
            if not resolved_path.is_file():
                raise FileNotFoundError(f"file `{resolved_path}` not found!")

            self.lib_path = os.fspath(resolved_path)
            self.lib = self.ffi.dlopen(self.lib_path)

        self.nq = self._get_c_array_len(self.lib.config_names)  # type: ignore[attr-defined]
        self.nv = self._get_c_array_len(self.lib.vel_names)  # type: ignore[attr-defined]
        self.nx = self.nq + self.nv
        self.nu = self.nv
        self.bodies_count = self._get_c_array_len(  # type: ignore[attr-defined]
            self.lib.kinematics_bodies
        )

    @cached_property
    def kinematics_size(self) -> int:
        return 7 * self.bodies_count if "quat" in self.lib_path else 3 * self.bodies_count

    def _get_c_array_len(self, ptr: object) -> int:
        count = 0
        while ptr[count] != self.ffi.NULL:  # type: ignore[index]
            count += 1
        return count

    @staticmethod
    def build_x_lib(lowstate: LowState_, odom: Odom) -> npt.NDArray[np.float64]:
        motor_state = lowstate.motor_state

        position = np.array([odom.position.x, odom.position.y, odom.position.z], dtype=np.float64)
        quat = np.array(
            [odom.quaternion.w, odom.quaternion.x, odom.quaternion.y, odom.quaternion.z],
            dtype=np.float64,
        )

        q_0_14 = np.array([motor.q for motor in motor_state[0:15]], dtype=np.float64)
        q_16_22 = np.array([motor.q for motor in motor_state[16:23]], dtype=np.float64)
        dq_0_14 = np.array([motor.dq for motor in motor_state[0:15]], dtype=np.float64)
        dq_16_22 = np.array([motor.dq for motor in motor_state[16:23]], dtype=np.float64)

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
