from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Iterable

import numpy as np
import numpy.typing as npt
from cffi import FFI

from g7_openarm_utils.gripper import (
    gripper_openness_to_model_position,
    gripper_openness_velocity_to_model_velocity,
)
from g7_openarm_utils.joint_layout import (
    ACTUATED_MODEL_JOINT_NAMES,
    FLOATING_BASE_CONFIG_NAMES,
    FLOATING_BASE_VELOCITY_NAMES,
    LEFT_GRIPPER_MOTOR_NAME,
    MODEL_JOINTS_BY_MOTOR_NAME,
    MOTOR_NAMES,
    RIGHT_GRIPPER_MOTOR_NAME,
    motor_index,
)
from g7_openarm_utils.quat import quat_to_rotation_matrix

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
        self.kinematics_bodies = self._read_c_string_array(  # type: ignore[attr-defined]
            self.lib.kinematics_bodies
        )

        self.q_index_by_name = MappingProxyType(
            self._unique_index_map(self.config_names, kind="configuration")
        )
        self.v_index_by_name = MappingProxyType(
            self._unique_index_map(self.vel_names, kind="velocity")
        )
        self.torque_index_by_name = MappingProxyType(
            self._unique_index_map(self.torque_names, kind="torque")
        )
        self.kinematics_body_index_by_name = MappingProxyType(
            self._unique_index_map(self.kinematics_bodies, kind="kinematics body")
        )

        self.nq = len(self.config_names)
        self.nv = len(self.vel_names)
        self.nx = self.nq + self.nv
        # Dynamics wrappers consume a full generalized-force vector of length nv.
        self.nu = self.nv
        self.nactuated = len(self.torque_names)
        self.bodies_count = len(self.kinematics_bodies)
        self.vector_order_api_version = self._vector_order_api_version()
        self.joint_names: tuple[str, ...] = ()
        self.joint_index_by_name = MappingProxyType({})

        self._validate_g7_layout()
        if self.vector_order_api_version >= 2:
            self.joint_names = self._read_c_string_array(self.lib.get_joint_names())
            self.joint_index_by_name = MappingProxyType(
                self._unique_index_map(self.joint_names, kind="joint")
            )
            self._validate_name_lookup_abi()

    def _vector_order_api_version(self) -> int:
        try:
            getter = self.lib.get_vector_order_api_version
        except AttributeError:
            # Legacy generated libraries expose the name arrays but no lookup API.
            return 1
        return int(getter())

    def _validate_name_lookup_abi(self) -> None:
        """Cross-check generated C name lookup functions against exported arrays."""
        reported_joint_count = int(self.lib.get_joint_count())
        if reported_joint_count != len(self.joint_names):
            raise RuntimeError(
                "PinnZoo joint metadata count mismatch: "
                f"get_joint_count()={reported_joint_count}, names={len(self.joint_names)}"
            )

        missing_joint_metadata = sorted(
            set(self._all_model_joint_names()) - set(self.joint_names)
        )
        if missing_joint_metadata:
            raise RuntimeError(
                "PinnZoo joint metadata is missing G7 model joints: "
                f"{missing_joint_metadata}"
            )

        for function_name, mapping in (
            ("get_config_index", self.q_index_by_name),
            ("get_vel_index", self.v_index_by_name),
            ("get_torque_index", self.torque_index_by_name),
        ):
            lookup = getattr(self.lib, function_name)
            for name, expected in mapping.items():
                actual = int(lookup(name.encode("utf-8")))
                if actual != expected:
                    raise RuntimeError(
                        f"PinnZoo {function_name}({name!r}) returned {actual}, "
                        f"expected {expected} from the exported order array"
                    )

        for joint_name in self._all_model_joint_names():
            q_index = int(self.lib.get_joint_q_index(joint_name.encode("utf-8")))
            v_index = int(self.lib.get_joint_v_index(joint_name.encode("utf-8")))
            nq = int(self.lib.get_joint_nq(joint_name.encode("utf-8")))
            nv = int(self.lib.get_joint_nv(joint_name.encode("utf-8")))
            expected_q = self.q_index(joint_name)
            expected_v = self.v_index(joint_name)
            if (q_index, v_index, nq, nv) != (expected_q, expected_v, 1, 1):
                raise RuntimeError(
                    f"PinnZoo joint metadata mismatch for {joint_name!r}: "
                    f"got q={q_index}, v={v_index}, nq={nq}, nv={nv}; "
                    f"expected q={expected_q}, v={expected_v}, nq=1, nv=1"
                )

    @cached_property
    def kinematics_body_size(self) -> int:
        if self.vector_order_api_version >= 2:
            size = int(self.lib.get_kinematics_body_size())
            if size <= 0:
                raise RuntimeError(f"invalid PinnZoo kinematics body size {size}")
            return size
        # Legacy libraries did not export representation metadata. Packaged G7
        # quaternion libraries retain ``quat`` in the filename.
        return 7 if "quat" in Path(self.lib_path).name else 3

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

    @staticmethod
    def _unique_index_map(names: tuple[str, ...], *, kind: str) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for index, name in enumerate(names):
            if not name:
                raise RuntimeError(f"PinnZoo {kind} name at index {index} is empty")
            if name in mapping:
                raise RuntimeError(
                    f"PinnZoo {kind} name {name!r} is duplicated at indices "
                    f"{mapping[name]} and {index}"
                )
            mapping[name] = index
        return mapping

    def _validate_g7_layout(self) -> None:
        scalar_joint_names = self._all_model_joint_names()
        expected_q = set((*FLOATING_BASE_CONFIG_NAMES, *scalar_joint_names))
        expected_v = set((*FLOATING_BASE_VELOCITY_NAMES, *scalar_joint_names))
        actual_q = set(self.config_names)
        actual_v = set(self.vel_names)
        if actual_q != expected_q or actual_v != expected_v:
            raise RuntimeError(
                "PinnZoo library layout does not exactly match the G7 model: "
                f"q missing={sorted(expected_q - actual_q)}, "
                f"q unexpected={sorted(actual_q - expected_q)}, "
                f"v missing={sorted(expected_v - actual_v)}, "
                f"v unexpected={sorted(actual_v - expected_v)}"
            )

        expected_torque = set(ACTUATED_MODEL_JOINT_NAMES)
        actual_torque = set(self.torque_names)
        if actual_torque != expected_torque:
            raise RuntimeError(
                "PinnZoo torque layout does not match the G7 actuated joints: "
                f"missing={sorted(expected_torque - actual_torque)}, "
                f"unexpected={sorted(actual_torque - expected_torque)}"
            )

        expected_bodies = {"L_tcp", "R_tcp"}
        actual_bodies = set(self.kinematics_bodies)
        if actual_bodies != expected_bodies:
            raise RuntimeError(
                "PinnZoo kinematics bodies do not match G7: "
                f"missing={sorted(expected_bodies - actual_bodies)}, "
                f"unexpected={sorted(actual_bodies - expected_bodies)}"
            )

    @staticmethod
    def _all_model_joint_names() -> tuple[str, ...]:
        return tuple(
            joint_name
            for motor_name in MOTOR_NAMES
            for joint_name in MODEL_JOINTS_BY_MOTOR_NAME[motor_name]
        )

    def q_index(self, name: str) -> int:
        try:
            return self.q_index_by_name[name]
        except KeyError as exc:
            raise KeyError(f"PinnZoo configuration name {name!r} not found") from exc

    def v_index(self, name: str) -> int:
        try:
            return self.v_index_by_name[name]
        except KeyError as exc:
            raise KeyError(f"PinnZoo velocity name {name!r} not found") from exc

    def torque_index(self, name: str) -> int:
        try:
            return self.torque_index_by_name[name]
        except KeyError as exc:
            raise KeyError(f"PinnZoo torque name {name!r} not found") from exc

    def q_indices(self, names: Iterable[str]) -> npt.NDArray[np.intp]:
        return np.asarray([self.q_index(name) for name in names], dtype=np.intp)

    def v_indices(self, names: Iterable[str]) -> npt.NDArray[np.intp]:
        return np.asarray([self.v_index(name) for name in names], dtype=np.intp)

    def torque_indices(self, names: Iterable[str]) -> npt.NDArray[np.intp]:
        return np.asarray([self.torque_index(name) for name in names], dtype=np.intp)

    def kinematics_body_index(self, name: str) -> int:
        try:
            return self.kinematics_body_index_by_name[name]
        except KeyError as exc:
            raise KeyError(f"PinnZoo kinematics body {name!r} not found") from exc

    def kinematics_pose_slice(self, name: str) -> slice:
        if self.kinematics_body_size != 7:
            raise RuntimeError("kinematics_pose_slice requires quaternion pose output")
        start = self.kinematics_body_size * self.kinematics_body_index(name)
        return slice(start, start + self.kinematics_body_size)

    def build_x_lib(self, lowstate: LowState_, odom: Odom) -> npt.NDArray[np.float64]:
        """Build the PinnZoo state strictly by exported q/v names, never by vector offsets."""
        q = np.zeros(self.nq, dtype=np.float64)
        v = np.zeros(self.nv, dtype=np.float64)

        base_q_values = (
            odom.position.x,
            odom.position.y,
            odom.position.z,
            odom.quaternion.w,
            odom.quaternion.x,
            odom.quaternion.y,
            odom.quaternion.z,
        )
        for name, value in zip(FLOATING_BASE_CONFIG_NAMES, base_q_values, strict=True):
            q[self.q_index(name)] = float(value)

        base_v_values = odom_velocity_world_to_body(odom)
        for name, value in zip(FLOATING_BASE_VELOCITY_NAMES, base_v_values, strict=True):
            v[self.v_index(name)] = float(value)

        for motor_name in MOTOR_NAMES:
            state = lowstate.motor_state[motor_index(motor_name)]
            q_value = float(state.q)
            v_value = float(state.dq)
            if motor_name in (LEFT_GRIPPER_MOTOR_NAME, RIGHT_GRIPPER_MOTOR_NAME):
                q_value = gripper_openness_to_model_position(q_value)
                v_value = gripper_openness_velocity_to_model_velocity(v_value)

            for joint_name in MODEL_JOINTS_BY_MOTOR_NAME[motor_name]:
                q[self.q_index(joint_name)] = q_value
                v[self.v_index(joint_name)] = v_value

        return np.concatenate((q, v))
