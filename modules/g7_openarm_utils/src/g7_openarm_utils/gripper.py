from __future__ import annotations

import math

GRIPPER_MODEL_OPEN_DISTANCE_M = 0.045
GRIPPER_MODEL_VELOCITY_LIMIT_M_S = 0.1
GRIPPER_OPENNESS_VELOCITY_LIMIT_PER_S = (
    GRIPPER_MODEL_VELOCITY_LIMIT_M_S / GRIPPER_MODEL_OPEN_DISTANCE_M
)


def _finite(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def _openness(value: float) -> float:
    value = _finite(value, name="openness")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"openness must be in [0, 1], got {value}")
    return value


def _motor_span(*, open_position: float, close_position: float) -> tuple[float, float, float]:
    open_position = _finite(open_position, name="open_position")
    close_position = _finite(close_position, name="close_position")
    span = open_position - close_position
    if span == 0.0:
        raise ValueError("open_position and close_position must differ")
    return open_position, close_position, span


def gripper_openness_to_model_position(openness: float) -> float:
    """Map normalized openness (0=closed, 1=open) to model joint position in meters."""
    return _openness(openness) * GRIPPER_MODEL_OPEN_DISTANCE_M


def gripper_openness_velocity_to_model_velocity(openness_velocity: float) -> float:
    """Map normalized openness velocity (positive=opening) to model velocity in m/s."""
    return _finite(openness_velocity, name="openness_velocity") * GRIPPER_MODEL_OPEN_DISTANCE_M


def gripper_model_position_to_openness(position: float) -> float:
    """Map model joint position in meters to normalized openness, clamped to [0, 1]."""
    position = _finite(position, name="position")
    openness = position / GRIPPER_MODEL_OPEN_DISTANCE_M
    return min(max(openness, 0.0), 1.0)


def gripper_model_velocity_to_openness_velocity(velocity: float) -> float:
    """Map model joint velocity in m/s to normalized openness velocity."""
    return _finite(velocity, name="velocity") / GRIPPER_MODEL_OPEN_DISTANCE_M


def gripper_openness_to_motor_position(
    openness: float,
    *,
    open_position: float,
    close_position: float,
) -> float:
    """Map normalized openness to a calibrated hardware motor position in radians."""
    openness = _openness(openness)
    _, close_position, span = _motor_span(
        open_position=open_position,
        close_position=close_position,
    )
    return close_position + span * openness


def gripper_openness_velocity_to_motor_velocity(
    openness_velocity: float,
    *,
    open_position: float,
    close_position: float,
) -> float:
    """Map normalized openness velocity to calibrated hardware motor velocity in rad/s."""
    openness_velocity = _finite(openness_velocity, name="openness_velocity")
    _, _, span = _motor_span(
        open_position=open_position,
        close_position=close_position,
    )
    return span * openness_velocity


def gripper_motor_position_to_openness(
    position: float,
    *,
    open_position: float,
    close_position: float,
) -> float:
    """Map calibrated hardware motor position in radians to normalized openness."""
    position = _finite(position, name="position")
    _, close_position, span = _motor_span(
        open_position=open_position,
        close_position=close_position,
    )
    openness = (position - close_position) / span
    return min(max(openness, 0.0), 1.0)


def gripper_motor_velocity_to_openness_velocity(
    velocity: float,
    *,
    open_position: float,
    close_position: float,
) -> float:
    """Map calibrated hardware motor velocity in rad/s to normalized openness velocity."""
    velocity = _finite(velocity, name="velocity")
    _, _, span = _motor_span(
        open_position=open_position,
        close_position=close_position,
    )
    return velocity / span
