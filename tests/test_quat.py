from __future__ import annotations

import numpy as np
import pytest

from g7_openarm_utils import quat_mul, quat_normalize, quat_rotate, quat_to_rotation_matrix


def test_quaternion_rotation_uses_wxyz_convention() -> None:
    half_angle = np.pi / 4.0
    quaternion = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])

    rotated = quat_rotate(quaternion, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-12)

    rotation = quat_to_rotation_matrix(quaternion)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)


def test_quat_mul_remains_a_raw_hamilton_product() -> None:
    # A Jacobian calculation may multiply non-unit quaternion-like vectors;
    # quat_mul must not silently normalize either operand.
    result = quat_mul([2.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result, [6.0, 0.0, 0.0, 0.0])


def test_quat_normalize_rejects_zero_quaternion() -> None:
    with pytest.raises(ValueError, match="norm"):
        quat_normalize([0.0, 0.0, 0.0, 0.0])
