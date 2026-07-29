from __future__ import annotations

import numpy as np
import pytest

from g7_openarm_wbc.stereographic_sew import (
    openarm_sew_cost_derivatives,
    openarm_sew_state,
    wrap_to_pi,
)


_WEIGHT = 4.0
_LEFT_TARGET = np.deg2rad(25.0)
_RIGHT_TARGET = np.deg2rad(-20.0)


def _x_lib() -> np.ndarray:
    x_lib = np.zeros(64, dtype=np.float64)
    x_lib[15:22] = np.array([0.2, 0.4, 0.3, -1.0, 0.2, -0.1, 0.1])
    x_lib[24:31] = np.array([-0.2, -0.4, -0.3, -1.0, -0.2, 0.1, -0.1])
    return x_lib


@pytest.mark.parametrize("base_enabled", [False, True])
def test_sew_cost_matches_scalar_cost_and_only_acts_on_arm_controls(
    base_enabled: bool,
) -> None:
    x_lib = _x_lib()

    cost, gradient, hessian = openarm_sew_cost_derivatives(
        x_lib[15:22],
        x_lib[24:31],
        left_target=_LEFT_TARGET,
        right_target=_RIGHT_TARGET,
        weight=_WEIGHT,
        base_enabled=base_enabled,
    )

    left = openarm_sew_state(x_lib[15:22], side="left")
    right = openarm_sew_state(x_lib[24:31], side="right")
    left_error = wrap_to_pi(_LEFT_TARGET - left.angle)
    right_error = wrap_to_pi(_RIGHT_TARGET - right.angle)
    expected_cost = 0.5 * _WEIGHT * (
        left_error * left_error + right_error * right_error
    )

    assert cost == pytest.approx(expected_cost)
    assert np.isfinite(gradient).all()
    assert np.isfinite(hessian).all()
    np.testing.assert_allclose(hessian, hessian.T, atol=1.0e-12)

    if base_enabled:
        np.testing.assert_allclose(gradient[:3], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(hessian[:3, :], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(hessian[:, :3], 0.0, atol=1.0e-12)

    assert np.linalg.norm(gradient) > 0.0
    assert np.linalg.norm(hessian) > 0.0


def test_sew_cost_is_zero_at_configured_targets() -> None:
    x_lib = _x_lib()
    left_target = openarm_sew_state(x_lib[15:22], side="left").angle
    right_target = openarm_sew_state(x_lib[24:31], side="right").angle

    cost, gradient, _ = openarm_sew_cost_derivatives(
        x_lib[15:22],
        x_lib[24:31],
        left_target=left_target,
        right_target=right_target,
        weight=_WEIGHT,
        base_enabled=False,
    )

    assert cost == pytest.approx(0.0, abs=1.0e-14)
    np.testing.assert_allclose(gradient, 0.0, atol=1.0e-12)
