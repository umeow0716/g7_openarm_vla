from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

ArmSide = Literal["left", "right"]

_EPSILON = 1.0e-9
_SHOULDER_Z = 0.132959535000066 + 0.689000000000002

_JOINT_AXES = (
    np.array([0.0, -1.0, 0.0], dtype=np.float64),
    np.array([1.0, 0.0, 0.0], dtype=np.float64),
    np.array([0.0, 0.0, -1.0], dtype=np.float64),
    np.array([0.0, 1.0, 0.0], dtype=np.float64),
    np.array([0.0, 0.0, -1.0], dtype=np.float64),
    np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    np.array([0.0, -1.0, 0.0], dtype=np.float64),
)

_JOINT_3_OFFSET = np.array([0.0284, 0.0, -0.051], dtype=np.float64)
_JOINT_5_OFFSET = np.array([0.0, 0.0, -0.0955], dtype=np.float64)
_JOINT_6_OFFSET = np.array([0.0, 0.0, -0.1205], dtype=np.float64)


class SEWSingularityError(ValueError):
    """Raised when the SEW angle is undefined at a geometric singularity."""


@dataclass(frozen=True, slots=True)
class SEWCartesianJacobian:
    """Partial derivatives of psi with respect to elbow and wrist positions."""

    elbow: npt.NDArray[np.float64]
    wrist: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OpenArmSEWPoints:
    """OpenArm shoulder, elbow and wrist points in the torso/body frame."""

    shoulder: npt.NDArray[np.float64]
    elbow: npt.NDArray[np.float64]
    wrist: npt.NDArray[np.float64]
    elbow_jacobian: npt.NDArray[np.float64]
    wrist_jacobian: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SEWState:
    angle: float
    jacobian: npt.NDArray[np.float64]
    points: OpenArmSEWPoints


def _vector3(value: npt.ArrayLike, *, name: str) -> npt.NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _normalize(value: npt.ArrayLike, *, name: str) -> npt.NDArray[np.float64]:
    vector = _vector3(value, name=name)
    norm = float(np.linalg.norm(vector))
    if norm < _EPSILON:
        raise SEWSingularityError(f"{name} has near-zero norm")
    return vector / norm


def _rotation_about_axis(axis: npt.NDArray[np.float64], angle: float) -> npt.NDArray[np.float64]:
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True, slots=True)
class StereographicSEW:
    """Stereographic shoulder-elbow-wrist angle parameterization.

    ``projection_pole`` is :math:`e_t`, the unit direction of the unavoidable
    half-line singularity. ``reference_direction`` is :math:`e_r`, a unit vector
    orthogonal to ``projection_pole``. Both vectors use the same frame as S, E and W.
    """

    projection_pole: npt.NDArray[np.float64]
    reference_direction: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        projection_pole = _normalize(self.projection_pole, name="projection_pole")
        reference_direction = _normalize(
            self.reference_direction,
            name="reference_direction",
        )
        dot = float(projection_pole @ reference_direction)
        if abs(dot) > 1.0e-8:
            raise ValueError(
                "projection_pole and reference_direction must be orthogonal, "
                f"got dot product {dot}"
            )

        object.__setattr__(self, "projection_pole", projection_pole)
        object.__setattr__(self, "reference_direction", reference_direction)

    def angle(
        self,
        shoulder: npt.ArrayLike,
        elbow: npt.ArrayLike,
        wrist: npt.ArrayLike,
    ) -> float:
        """Return stereographic SEW angle psi in radians."""
        shoulder_position = _vector3(shoulder, name="shoulder")
        elbow_vector = _vector3(elbow, name="elbow") - shoulder_position
        wrist_vector = _vector3(wrist, name="wrist") - shoulder_position
        shoulder_wrist_direction = _normalize(wrist_vector, name="wrist - shoulder")

        sew_plane_normal = _normalize(
            np.cross(wrist_vector, elbow_vector),
            name="SEW plane normal",
        )
        reference_plane_normal = _normalize(
            np.cross(
                shoulder_wrist_direction - self.projection_pole,
                self.reference_direction,
            ),
            name="stereographic reference plane normal",
        )

        sine = float(
            sew_plane_normal
            @ np.cross(shoulder_wrist_direction, reference_plane_normal)
        )
        cosine = float(np.clip(sew_plane_normal @ reference_plane_normal, -1.0, 1.0))
        return float(np.arctan2(sine, cosine))

    def cartesian_jacobian(
        self,
        shoulder: npt.ArrayLike,
        elbow: npt.ArrayLike,
        wrist: npt.ArrayLike,
    ) -> SEWCartesianJacobian:
        """Return analytic dpsi/dE and dpsi/dW for a fixed shoulder point."""
        shoulder_position = _vector3(shoulder, name="shoulder")
        elbow_vector = _vector3(elbow, name="elbow") - shoulder_position
        wrist_vector = _vector3(wrist, name="wrist") - shoulder_position

        wrist_norm = float(np.linalg.norm(wrist_vector))
        if wrist_norm < _EPSILON:
            raise SEWSingularityError("wrist and shoulder positions are coincident")
        shoulder_wrist_direction = wrist_vector / wrist_norm

        reference_plane_normal = np.cross(
            shoulder_wrist_direction - self.projection_pole,
            self.reference_direction,
        )
        x_c = np.cross(reference_plane_normal, wrist_vector)
        x_c_norm = float(np.linalg.norm(x_c))
        if x_c_norm < _EPSILON:
            raise SEWSingularityError(
                "stereographic reference is singular near the projection-pole half-line"
            )
        x_c_unit = x_c / x_c_norm
        y_c_unit = np.cross(shoulder_wrist_direction, x_c_unit)

        projected_elbow = (
            np.eye(3, dtype=np.float64)
            - np.outer(shoulder_wrist_direction, shoulder_wrist_direction)
        ) @ elbow_vector
        projected_elbow_norm = float(np.linalg.norm(projected_elbow))
        if projected_elbow_norm < _EPSILON:
            raise SEWSingularityError("shoulder, elbow and wrist positions are collinear")
        projected_elbow_unit = projected_elbow / projected_elbow_norm

        common = np.cross(shoulder_wrist_direction, projected_elbow_unit)
        elbow_jacobian = common / projected_elbow_norm

        wrist_term_1 = (
            float(shoulder_wrist_direction @ self.reference_direction)
            / x_c_norm
            * y_c_unit
        )
        wrist_term_2 = (
            float(
                shoulder_wrist_direction
                @ np.cross(self.projection_pole, self.reference_direction)
            )
            / x_c_norm
            * x_c_unit
        )
        wrist_term_3 = (
            float(shoulder_wrist_direction @ elbow_vector)
            / wrist_norm
            / projected_elbow_norm
            * common
        )
        wrist_jacobian = wrist_term_1 + wrist_term_2 - wrist_term_3

        return SEWCartesianJacobian(
            elbow=elbow_jacobian,
            wrist=wrist_jacobian,
        )

    def joint_jacobian(
        self,
        shoulder: npt.ArrayLike,
        elbow: npt.ArrayLike,
        wrist: npt.ArrayLike,
        elbow_jacobian: npt.ArrayLike,
        wrist_jacobian: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Apply J_psi = J_psi,E J_E + J_psi,W J_W."""
        elbow_position_jacobian = np.asarray(elbow_jacobian, dtype=np.float64)
        wrist_position_jacobian = np.asarray(wrist_jacobian, dtype=np.float64)
        if elbow_position_jacobian.ndim != 2 or elbow_position_jacobian.shape[0] != 3:
            raise ValueError(
                "elbow_jacobian must have shape (3, n), "
                f"got {elbow_position_jacobian.shape}"
            )
        if wrist_position_jacobian.shape != elbow_position_jacobian.shape:
            raise ValueError(
                "wrist_jacobian must have the same shape as elbow_jacobian, "
                f"got {wrist_position_jacobian.shape} and {elbow_position_jacobian.shape}"
            )

        cartesian = self.cartesian_jacobian(shoulder, elbow, wrist)
        return cartesian.elbow @ elbow_position_jacobian + cartesian.wrist @ wrist_position_jacobian


def _openarm_geometry(
    side: ArmSide,
) -> tuple[
    npt.NDArray[np.float64],
    tuple[npt.NDArray[np.float64], ...],
]:
    if side == "left":
        side_sign = 1.0
        joint_4_y = -5.0e-5
    elif side == "right":
        side_sign = -1.0
        joint_4_y = 0.0
    else:
        raise ValueError(f"Unsupported arm side: {side}")

    shoulder = np.array([0.0, side_sign * 0.0875, _SHOULDER_Z], dtype=np.float64)
    offsets = (
        np.array([-0.0284, side_sign * 0.065, 0.0], dtype=np.float64),
        _JOINT_3_OFFSET,
        np.array([0.0, joint_4_y, -0.169], dtype=np.float64),
        _JOINT_5_OFFSET,
        _JOINT_6_OFFSET,
    )
    return shoulder, offsets


def openarm_sew_points(q: npt.ArrayLike, *, side: ArmSide) -> OpenArmSEWPoints:
    """Return OpenArm S/E/W points and their position Jacobians in torso frame.

    S is the J1 origin, E is the J4 origin, and W is the J6 origin. The J6 origin
    lies on the J5 axis, so the selected wrist point is independent of wrist roll.
    """
    joint_angles = np.asarray(q, dtype=np.float64)
    if joint_angles.shape != (7,):
        raise ValueError(f"q must have shape (7,), got {joint_angles.shape}")
    if not np.isfinite(joint_angles).all():
        raise ValueError("q must contain only finite values")

    shoulder, offsets = _openarm_geometry(side)
    position = shoulder.copy()
    rotation = np.eye(3, dtype=np.float64)
    joint_origins: list[npt.NDArray[np.float64]] = []
    joint_axes: list[npt.NDArray[np.float64]] = []

    elbow: npt.NDArray[np.float64] | None = None
    wrist: npt.NDArray[np.float64] | None = None

    for joint_index, (axis, angle, child_offset) in enumerate(
        zip(_JOINT_AXES[:5], joint_angles[:5], offsets, strict=True)
    ):
        joint_origins.append(position.copy())
        joint_axes.append(rotation @ axis)
        rotation = rotation @ _rotation_about_axis(axis, float(angle))
        position = position + rotation @ child_offset

        if joint_index == 2:
            elbow = position.copy()
        elif joint_index == 4:
            wrist = position.copy()

    assert elbow is not None
    assert wrist is not None

    elbow_jacobian = np.zeros((3, 7), dtype=np.float64)
    wrist_jacobian = np.zeros((3, 7), dtype=np.float64)
    for joint_index, (joint_origin, joint_axis) in enumerate(
        zip(joint_origins, joint_axes, strict=True)
    ):
        if joint_index <= 2:
            elbow_jacobian[:, joint_index] = np.cross(
                joint_axis,
                elbow - joint_origin,
            )
        wrist_jacobian[:, joint_index] = np.cross(
            joint_axis,
            wrist - joint_origin,
        )

    return OpenArmSEWPoints(
        shoulder=shoulder,
        elbow=elbow,
        wrist=wrist,
        elbow_jacobian=elbow_jacobian,
        wrist_jacobian=wrist_jacobian,
    )


def openarm_stereographic_sew(*, side: ArmSide) -> StereographicSEW:
    """Return a torso-frame parameterization with its singular half-line below the robot."""
    if side == "left":
        outward = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    elif side == "right":
        outward = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported arm side: {side}")

    return StereographicSEW(
        projection_pole=np.array([0.0, 0.0, -1.0], dtype=np.float64),
        reference_direction=outward,
    )


def openarm_sew_state(
    q: npt.ArrayLike,
    *,
    side: ArmSide,
    parameterization: StereographicSEW | None = None,
) -> SEWState:
    """Return the current OpenArm stereographic SEW angle and dpsi/dq."""
    points = openarm_sew_points(q, side=side)
    sew = openarm_stereographic_sew(side=side) if parameterization is None else parameterization
    angle = sew.angle(points.shoulder, points.elbow, points.wrist)
    jacobian = sew.joint_jacobian(
        points.shoulder,
        points.elbow,
        points.wrist,
        points.elbow_jacobian,
        points.wrist_jacobian,
    )
    return SEWState(angle=angle, jacobian=jacobian, points=points)


def openarm_sew_cost_derivatives(
    left_q: npt.ArrayLike,
    right_q: npt.ArrayLike,
    *,
    left_target: float,
    right_target: float,
    weight: float,
    base_enabled: bool,
) -> tuple[
    float,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Return bilateral SEW cost and Gauss-Newton derivatives.

    The targets and current SEW angles are in radians. The returned gradient and
    Hessian follow the WBC control layout: optional base controls, seven left-arm
    controls, then seven right-arm controls.
    """
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(f"weight must be finite and non-negative, got {weight}")

    control_size = 17 if base_enabled else 14
    arm_offset = 3 if base_enabled else 0
    cost = 0.0
    gradient = np.zeros(control_size, dtype=np.float64)
    hessian = np.zeros((control_size, control_size), dtype=np.float64)

    for side, q, target, control_offset in (
        ("left", left_q, left_target, arm_offset),
        ("right", right_q, right_target, arm_offset + 7),
    ):
        try:
            sew_state = openarm_sew_state(q, side=side)
        except SEWSingularityError:
            continue

        error = wrap_to_pi(float(target) - sew_state.angle)
        error_jacobian = np.zeros(control_size, dtype=np.float64)
        error_jacobian[control_offset : control_offset + 7] = -sew_state.jacobian

        cost += 0.5 * weight * error * error
        gradient += weight * error_jacobian * error
        hessian += weight * np.outer(error_jacobian, error_jacobian)

    return cost, gradient, hessian
