import numpy as np
import pytest

from src.monitoring.psi_monitor import calculate_psi


def test_calculate_psi_zero_for_equal_distributions():
    base = np.linspace(0.1, 0.9, 100)
    psi = calculate_psi(base, base)
    assert psi == pytest.approx(0.0)


def test_calculate_psi_positive_for_shift():
    expected = np.linspace(0.1, 0.9, 100)
    actual = np.linspace(0.2, 0.95, 100)
    psi = calculate_psi(expected, actual)
    assert psi > 0
