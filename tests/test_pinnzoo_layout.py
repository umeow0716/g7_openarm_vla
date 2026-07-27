from __future__ import annotations

import numpy as np

from g7_openarm_pinnzoo import PinnZooModel, forward_dynamics, forward_dynamics_deriv, zero_state


def test_packaged_native_library_and_derivative_layout() -> None:
    model = PinnZooModel()
    x = zero_state(model)
    x[3] = 1.0
    tau = np.linspace(-0.2, 0.2, model.nv, dtype=np.float64)

    derivative_x, derivative_tau = forward_dynamics_deriv(model, x, tau)

    rng = np.random.default_rng(7)
    direction_x = rng.normal(size=model.nx)
    direction_tau = rng.normal(size=model.nv)
    epsilon = 1e-6

    finite_x = (
        forward_dynamics(model, x + epsilon * direction_x, tau)
        - forward_dynamics(model, x - epsilon * direction_x, tau)
    ) / (2.0 * epsilon)
    finite_tau = (
        forward_dynamics(model, x, tau + epsilon * direction_tau)
        - forward_dynamics(model, x, tau - epsilon * direction_tau)
    ) / (2.0 * epsilon)

    np.testing.assert_allclose(derivative_x @ direction_x, finite_x, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(
        derivative_tau @ direction_tau,
        finite_tau,
        rtol=1e-5,
        atol=1e-7,
    )
