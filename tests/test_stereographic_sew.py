from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from g7_openarm_wbc.stereographic_sew import (
    ArmSide,
    SEWSingularityError,
    StereographicSEW,
    openarm_sew_points,
    openarm_sew_state,
    wrap_to_pi,
)


def _finite_difference_angle(
    sew: StereographicSEW,
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    *,
    point: str,
    epsilon: float = 1.0e-7,
) -> np.ndarray:
    jacobian = np.zeros(3, dtype=np.float64)
    for index in range(3):
        delta = np.zeros(3, dtype=np.float64)
        delta[index] = epsilon
        if point == "elbow":
            positive = sew.angle(shoulder, elbow + delta, wrist)
            negative = sew.angle(shoulder, elbow - delta, wrist)
        elif point == "wrist":
            positive = sew.angle(shoulder, elbow, wrist + delta)
            negative = sew.angle(shoulder, elbow, wrist - delta)
        else:
            raise ValueError(f"Unsupported point: {point}")
        jacobian[index] = wrap_to_pi(positive - negative) / (2.0 * epsilon)
    return jacobian


def test_stereographic_cartesian_jacobian_matches_finite_difference() -> None:
    sew = StereographicSEW(
        projection_pole=np.array([0.0, 0.0, -1.0]),
        reference_direction=np.array([0.0, 1.0, 0.0]),
    )
    shoulder = np.array([0.1, -0.2, 0.3])
    elbow = np.array([0.35, 0.1, 0.15])
    wrist = np.array([0.55, -0.05, 0.5])

    analytic = sew.cartesian_jacobian(shoulder, elbow, wrist)
    elbow_numeric = _finite_difference_angle(
        sew,
        shoulder,
        elbow,
        wrist,
        point="elbow",
    )
    wrist_numeric = _finite_difference_angle(
        sew,
        shoulder,
        elbow,
        wrist,
        point="wrist",
    )

    np.testing.assert_allclose(analytic.elbow, elbow_numeric, atol=1.0e-7, rtol=1.0e-7)
    np.testing.assert_allclose(analytic.wrist, wrist_numeric, atol=1.0e-7, rtol=1.0e-7)


@pytest.mark.parametrize(
    ("side", "q"),
    [
        ("left", np.array([0.2, 0.4, 0.3, -1.0, 0.2, -0.1, 0.1])),
        ("right", np.array([-0.2, -0.4, -0.3, -1.0, -0.2, 0.1, -0.1])),
    ],
)
def test_openarm_point_jacobians_match_finite_difference(
    side: str,
    q: np.ndarray,
) -> None:
    arm_side = cast(ArmSide, side)
    analytic = openarm_sew_points(q, side=arm_side)
    epsilon = 1.0e-7
    elbow_numeric = np.zeros((3, 7), dtype=np.float64)
    wrist_numeric = np.zeros((3, 7), dtype=np.float64)

    for index in range(7):
        delta = np.zeros(7, dtype=np.float64)
        delta[index] = epsilon
        positive = openarm_sew_points(q + delta, side=arm_side)
        negative = openarm_sew_points(q - delta, side=arm_side)
        elbow_numeric[:, index] = (positive.elbow - negative.elbow) / (2.0 * epsilon)
        wrist_numeric[:, index] = (positive.wrist - negative.wrist) / (2.0 * epsilon)

    np.testing.assert_allclose(
        analytic.elbow_jacobian,
        elbow_numeric,
        atol=1.0e-8,
        rtol=1.0e-7,
    )
    np.testing.assert_allclose(
        analytic.wrist_jacobian,
        wrist_numeric,
        atol=1.0e-8,
        rtol=1.0e-7,
    )


@pytest.mark.parametrize(
    ("side", "q"),
    [
        ("left", np.array([0.2, 0.4, 0.3, -1.0, 0.2, -0.1, 0.1])),
        ("right", np.array([-0.2, -0.4, -0.3, -1.0, -0.2, 0.1, -0.1])),
    ],
)
def test_openarm_sew_joint_jacobian_matches_finite_difference(
    side: str,
    q: np.ndarray,
) -> None:
    arm_side = cast(ArmSide, side)
    state = openarm_sew_state(q, side=arm_side)
    epsilon = 1.0e-7
    numeric = np.zeros(7, dtype=np.float64)

    for index in range(7):
        delta = np.zeros(7, dtype=np.float64)
        delta[index] = epsilon
        positive = openarm_sew_state(q + delta, side=arm_side).angle
        negative = openarm_sew_state(q - delta, side=arm_side).angle
        numeric[index] = wrap_to_pi(positive - negative) / (2.0 * epsilon)

    np.testing.assert_allclose(state.jacobian, numeric, atol=1.0e-7, rtol=1.0e-7)
    np.testing.assert_allclose(state.jacobian[4:], 0.0, atol=1.0e-12)


def test_stereographic_sew_rejects_invalid_reference_pair() -> None:
    with pytest.raises(ValueError, match="orthogonal"):
        StereographicSEW(
            projection_pole=np.array([0.0, 0.0, -1.0]),
            reference_direction=np.array([0.0, 1.0, -1.0]),
        )


def test_stereographic_sew_reports_geometric_singularities() -> None:
    sew = StereographicSEW(
        projection_pole=np.array([0.0, 0.0, -1.0]),
        reference_direction=np.array([0.0, 1.0, 0.0]),
    )

    with pytest.raises(SEWSingularityError, match="collinear|SEW plane normal"):
        sew.angle(
            np.zeros(3),
            np.array([0.0, 0.0, 0.5]),
            np.array([0.0, 0.0, 1.0]),
        )
