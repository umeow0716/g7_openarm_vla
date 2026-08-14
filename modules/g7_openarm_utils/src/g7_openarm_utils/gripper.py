from __future__ import annotations

import math

GRIPPER_COMMAND_RANGE = 0.45


def _finite(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def gripper_openness_to_command(openness: float) -> float:
    """Convert normalized openness (0=closed, 1=open) to lowcmd coordinates."""
    openness = _finite(openness, name="openness")
    if not 0.0 <= openness <= 1.0:
        raise ValueError(f"openness must be in [0, 1], got {openness}")
    return (1.0 - openness) * GRIPPER_COMMAND_RANGE


def gripper_command_to_openness(command: float) -> float:
    """Convert the legacy lowcmd gripper coordinate to normalized openness."""
    command = _finite(command, name="command")
    return 1.0 - command / GRIPPER_COMMAND_RANGE


def gripper_command_to_motor_position(
    command: float,
    *,
    open_position: float,
    close_position: float,
) -> float:
    """Map a lowcmd gripper coordinate to a calibrated hardware motor position."""
    command = _finite(command, name="command")
    open_position = _finite(open_position, name="open_position")
    close_position = _finite(close_position, name="close_position")
    return (
        (close_position - open_position) * command / GRIPPER_COMMAND_RANGE
        + open_position
    )


def gripper_command_velocity_to_motor_velocity(
    velocity: float,
    *,
    open_position: float,
    close_position: float,
) -> float:
    """Convert lowcmd-coordinate velocity to calibrated hardware motor velocity."""
    velocity = _finite(velocity, name="velocity")
    open_position = _finite(open_position, name="open_position")
    close_position = _finite(close_position, name="close_position")
    return velocity * (close_position - open_position) / GRIPPER_COMMAND_RANGE


def gripper_motor_position_to_command(
    position: float,
    *,
    open_position: float,
    close_position: float,
) -> float:
    """Map a calibrated hardware motor position back to the lowcmd coordinate."""
    position = _finite(position, name="position")
    open_position = _finite(open_position, name="open_position")
    close_position = _finite(close_position, name="close_position")
    span = close_position - open_position
    if span == 0.0:
        raise ValueError("open_position and close_position must differ")
    return (position - open_position) * GRIPPER_COMMAND_RANGE / span


def gripper_motor_velocity_to_command_velocity(
    velocity: float,
    *,
    open_position: float,
    close_position: float,
) -> float:
    """Convert calibrated hardware motor velocity to lowcmd-coordinate velocity."""
    velocity = _finite(velocity, name="velocity")
    open_position = _finite(open_position, name="open_position")
    close_position = _finite(close_position, name="close_position")
    span = close_position - open_position
    if span == 0.0:
        raise ValueError("open_position and close_position must differ")
    return velocity * GRIPPER_COMMAND_RANGE / span
