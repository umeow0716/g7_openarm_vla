from __future__ import annotations

import os
import numpy as np
import numpy.typing as npt
import platform

from cffi import FFI
from collections.abc import Generator
from contextlib import contextmanager
from functools import cached_property
from g7_openarm_utils.quat import quat_to_rotation_matrix
from importlib.resources import as_file, files
from pathlib import Path
from typing import  TYPE_CHECKING

if TYPE_CHECKING:
    from g7_openarm_idl import Odom


class NotSupportedArchitecture(Exception):
    pass


def get_arch():
    machine = platform.machine().lower()

    if machine in ("x86_64", "amd64"):
        return "x86_64"
    elif machine in ("aarch64", "arm64"):
        return "aarch64"

    raise NotSupportedArchitecture(f"{platform.machine().lower()} is not support")

@contextmanager
def native_library_path() -> Generator[Path]:
    resource = files("g7_openarm_pinnzoo").joinpath(
        "lib",
        f"libg7_openarm_quat_{get_arch()}.so",
    )

    if not resource.is_file():
        raise FileNotFoundError(f"Packaged PinnZoo library not found: {resource}")

    with as_file(resource) as path:
        yield path


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
int get_vector_order_api_version(void);
int get_kinematics_body_size(void);
int get_config_index(const char* name);
int get_vel_index(const char* name);
int get_torque_index(const char* name);
const char** get_joint_names(void);
int get_joint_count(void);
int get_joint_q_index(const char* name);
int get_joint_v_index(const char* name);
int get_joint_nq(const char* name);
int get_joint_nv(const char* name);
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

        self.config_names = self._read_c_string_array(
            self.lib.config_names  # type: ignore[attr-defined]
        )
        self.vel_names = self._read_c_string_array(self.lib.vel_names)  # type: ignore[attr-defined]
        self.torque_names = self._read_c_string_array(
            self.lib.torque_names  # type: ignore[attr-defined]
        )
        self.kinematics_bodies = self._read_c_string_array(
            self.lib.kinematics_bodies  # type: ignore[attr-defined]
        )
        self.bodies_count = len(self.kinematics_bodies)

        self.nq = len(self.config_names)
        self.nv = len(self.vel_names)
        self.nx = self.nq + self.nv
        self.nu = self.nv

    @cached_property
    def kinematics_body_size(self) -> int:
        size = int(self.lib.get_kinematics_body_size()) # type: ignore
        if size <= 0:
            raise RuntimeError(f"invalid PinnZoo kinematics body size {size}")
        return size

    @cached_property
    def kinematics_size(self) -> int:
        return self.kinematics_body_size * self.bodies_count

    def _read_c_string_array(self, ptr: object) -> tuple[str, ...]:
        values: list[str] = []
        index = 0
        while ptr[index] != self.ffi.NULL:  # type: ignore[index]
            values.append(self.ffi.string(ptr[index]).decode("utf-8"))  # type: ignore[index]
            index += 1
        return tuple(values)
