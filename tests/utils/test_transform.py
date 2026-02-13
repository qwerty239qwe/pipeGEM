"""Tests for pipeGEM/utils/transform.py — sigmoid, mod_log10, exp_x."""
import numpy as np
import pytest

from pipeGEM.utils.transform import sigmoid, mod_log10, exp_x, functions


class TestSigmoid:

    def test_sigmoid_at_zero(self):
        """sigmoid(0, any_coef) = 0.5."""
        assert np.isclose(sigmoid(0, 1.0), 0.5)
        assert np.isclose(sigmoid(0, 5.0), 0.5)
        assert np.isclose(sigmoid(0, -3.0), 0.5)

    def test_sigmoid_monotonic(self):
        """Increasing x -> increasing sigmoid for positive coef."""
        x_vals = np.linspace(-5, 5, 100)
        y_vals = sigmoid(x_vals, coef=2.0)
        assert np.all(np.diff(y_vals) > 0)

    def test_sigmoid_bounds(self):
        """Output in [0, 1]."""
        x_vals = np.linspace(-100, 100, 1000)
        y_vals = sigmoid(x_vals, coef=1.0)
        assert np.all(y_vals >= 0)
        assert np.all(y_vals <= 1)
        # For moderate range, strictly between 0 and 1
        y_moderate = sigmoid(np.linspace(-5, 5, 100), coef=1.0)
        assert np.all(y_moderate > 0)
        assert np.all(y_moderate < 1)


class TestModLog10:

    def test_mod_log10_known(self):
        """mod_log10(0, 1001) = log10(1001)."""
        expected = np.log10(1001)
        assert np.isclose(mod_log10(0, 1001), expected)

    def test_mod_log10_positive_input(self):
        """mod_log10(99, 1001) = log10(99 + 1001) = log10(1100)."""
        assert np.isclose(mod_log10(99, 1001), np.log10(1100))

    def test_mod_log10_default_coef(self):
        """Default coef=1001: mod_log10(0) = log10(1001)."""
        assert np.isclose(mod_log10(0), np.log10(1001))


class TestExpX:

    def test_exp_x_at_zero(self):
        """exp_x(0, 1, 1.1) = 1.1^0 - 1 = 0.0."""
        assert np.isclose(exp_x(0, coef=1, a=1.1), 0.0)

    def test_exp_x_positive(self):
        """exp_x(1, 1, 2) = 2^1 - 1 = 1.0."""
        assert np.isclose(exp_x(1, coef=1, a=2), 1.0)

    def test_exp_x_at_zero_default(self):
        """exp_x(0) with defaults (coef=1, a=1.1) = 0."""
        assert np.isclose(exp_x(0), 0.0)


class TestFunctionsDict:

    def test_functions_has_keys(self):
        assert "sigmoid" in functions
        assert "mod_log10" in functions
        assert "exp_x" in functions

    def test_functions_values_callable(self):
        for name, fn in functions.items():
            assert callable(fn), f"{name} is not callable"
