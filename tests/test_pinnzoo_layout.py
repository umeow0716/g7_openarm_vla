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


def test_packaged_library_exports_semantic_g7_layout() -> None:
    from g7_openarm_utils import (
        ACTUATED_MODEL_JOINT_NAMES,
        FLOATING_BASE_CONFIG_NAMES,
        FLOATING_BASE_VELOCITY_NAMES,
        MODEL_JOINTS_BY_MOTOR_NAME,
        MOTOR_NAMES,
    )

    model = PinnZooModel()
    scalar_joint_names = tuple(
        joint_name
        for motor_name in MOTOR_NAMES
        for joint_name in MODEL_JOINTS_BY_MOTOR_NAME[motor_name]
    )

    assert model.config_names[:7] == FLOATING_BASE_CONFIG_NAMES
    assert model.vel_names[:6] == FLOATING_BASE_VELOCITY_NAMES
    assert set(scalar_joint_names) <= set(model.config_names)
    assert set(scalar_joint_names) <= set(model.vel_names)
    assert set(model.torque_names) == set(ACTUATED_MODEL_JOINT_NAMES)
    assert model.kinematics_bodies == ("L_tcp", "R_tcp")
    assert model.kinematics_pose_slice("L_tcp") == slice(0, 7)
    assert model.kinematics_pose_slice("R_tcp") == slice(7, 14)

    # Free-flyer q and v have different dimensionality. This assertion prevents
    # future code from treating q-index and v-index as interchangeable.
    assert model.q_index("L_1_joint") != model.v_index("L_1_joint")


def test_build_x_lib_resolves_independent_permuted_q_and_v_name_orders() -> None:
    from types import MappingProxyType, SimpleNamespace

    from g7_openarm_utils import (
        FLOATING_BASE_CONFIG_NAMES,
        FLOATING_BASE_VELOCITY_NAMES,
        LEFT_GRIPPER_MOTOR_NAME,
        MODEL_JOINTS_BY_MOTOR_NAME,
        MOTOR_NAMES,
        RIGHT_GRIPPER_MOTOR_NAME,
        gripper_openness_to_model_position,
        gripper_openness_velocity_to_model_velocity,
        motor_index,
    )

    scalar_joint_names = tuple(
        joint_name
        for motor_name in MOTOR_NAMES
        for joint_name in MODEL_JOINTS_BY_MOTOR_NAME[motor_name]
    )
    q_names = tuple(reversed((*FLOATING_BASE_CONFIG_NAMES, *scalar_joint_names)))
    v_names = (*scalar_joint_names[9:], *FLOATING_BASE_VELOCITY_NAMES, *scalar_joint_names[:9])

    model = object.__new__(PinnZooModel)
    model.nq = len(q_names)
    model.nv = len(v_names)
    model.q_index_by_name = MappingProxyType({name: i for i, name in enumerate(q_names)})
    model.v_index_by_name = MappingProxyType({name: i for i, name in enumerate(v_names)})

    lowstate = SimpleNamespace(
        motor_state=[
            SimpleNamespace(q=0.1 * (index + 1), dq=-0.2 * (index + 1))
            for index in range(len(MOTOR_NAMES))
        ]
    )
    lowstate.motor_state[motor_index(LEFT_GRIPPER_MOTOR_NAME)].q = 0.25
    lowstate.motor_state[motor_index(LEFT_GRIPPER_MOTOR_NAME)].dq = 0.5
    lowstate.motor_state[motor_index(RIGHT_GRIPPER_MOTOR_NAME)].q = 0.75
    lowstate.motor_state[motor_index(RIGHT_GRIPPER_MOTOR_NAME)].dq = -0.5
    odom = SimpleNamespace(
        position=SimpleNamespace(x=1.1, y=2.2, z=3.3),
        quaternion=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
        velocity=SimpleNamespace(x=4.4, y=5.5, z=6.6),
        angular_velocity=SimpleNamespace(x=7.7, y=8.8, z=9.9),
    )

    x = model.build_x_lib(lowstate, odom)
    q = x[: model.nq]
    v = x[model.nq :]

    for name, expected in zip(
        FLOATING_BASE_CONFIG_NAMES,
        (1.1, 2.2, 3.3, 1.0, 0.0, 0.0, 0.0),
        strict=True,
    ):
        assert q[model.q_index(name)] == expected
    for name, expected in zip(
        FLOATING_BASE_VELOCITY_NAMES,
        (4.4, 5.5, 6.6, 7.7, 8.8, 9.9),
        strict=True,
    ):
        assert v[model.v_index(name)] == expected

    for motor_name in MOTOR_NAMES:
        state = lowstate.motor_state[motor_index(motor_name)]
        expected_q = state.q
        expected_v = state.dq
        if motor_name in (LEFT_GRIPPER_MOTOR_NAME, RIGHT_GRIPPER_MOTOR_NAME):
            expected_q = gripper_openness_to_model_position(expected_q)
            expected_v = gripper_openness_velocity_to_model_velocity(expected_v)
        for joint_name in MODEL_JOINTS_BY_MOTOR_NAME[motor_name]:
            assert q[model.q_index(joint_name)] == expected_q
            assert v[model.v_index(joint_name)] == expected_v
