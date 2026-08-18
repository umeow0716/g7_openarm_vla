from __future__ import annotations

import numpy as np

from g7_openarm_utils import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES
from g7_openarm_wbc import ik_solver
from g7_openarm_wbc.control_layout import ARM_CONTROL_NAMES, control_indices


class _PermutedModel:
    def __init__(self) -> None:
        all_names = (
            "x",
            "y",
            "z",
            "q_w",
            "q_x",
            "q_y",
            "q_z",
            *LEFT_ARM_JOINT_NAMES,
            *RIGHT_ARM_JOINT_NAMES,
        )
        # Intentionally non-contiguous, non-URDF order indices.
        self.mapping = {name: 2 * index + 1 for index, name in enumerate(reversed(all_names))}

    def q_indices(self, names):
        return np.asarray([self.mapping[name] for name in names], dtype=np.intp)


def test_task_jacobian_selects_arm_columns_by_name(monkeypatch) -> None:
    model = _PermutedModel()
    width = max(model.mapping.values()) + 2
    raw = np.tile(np.arange(width, dtype=np.float64), (4, 1))
    monkeypatch.setattr(ik_solver, "kinematics_jacobian", lambda _model, _x: raw)

    result = ik_solver.task_kinematic_jacobian(
        model,
        np.zeros(width, dtype=np.float64),
        base_enabled=False,
    )
    expected_indices = [
        *[model.mapping[name] for name in LEFT_ARM_JOINT_NAMES],
        *[model.mapping[name] for name in RIGHT_ARM_JOINT_NAMES],
    ]
    np.testing.assert_array_equal(result, raw[:, expected_indices])


def test_control_indices_are_cached_after_name_resolution() -> None:
    first = control_indices(ARM_CONTROL_NAMES, base_enabled=True)
    second = control_indices(ARM_CONTROL_NAMES, base_enabled=True)

    assert first is second
    assert not first.flags.writeable
