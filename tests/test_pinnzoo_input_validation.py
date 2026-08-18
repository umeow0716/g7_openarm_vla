from __future__ import annotations

import numpy as np
import pytest

from g7_openarm_pinnzoo.pinnzoo_func import _vector_input


def test_pinnzoo_vector_input_requires_exact_one_dimensional_shape() -> None:
    with pytest.raises(ValueError, match="Expected x shape"):
        _vector_input("x", np.zeros((4, 1), dtype=np.float64), 4)

    with pytest.raises(ValueError, match="Expected x shape"):
        _vector_input("x", np.zeros(3, dtype=np.float64), 4)


def test_pinnzoo_vector_input_normalizes_dtype_and_memory_layout() -> None:
    source = np.arange(8, dtype=np.float32)[::2]
    result = _vector_input("x", source, 4)

    assert result.dtype == np.float64
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, [0.0, 2.0, 4.0, 6.0])


def test_pinnzoo_vector_input_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _vector_input("tau", np.array([0.0, np.nan], dtype=np.float64), 2)
