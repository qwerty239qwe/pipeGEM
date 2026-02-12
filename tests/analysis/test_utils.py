"""Tests for pipeGEM/analysis/_utils.py — p-value correction functions."""
import numpy as np
import pytest

from pipeGEM.analysis._utils import (
    bh_adjust,
    bonferroni_adjust,
    holm_adjust,
    adjust_p_values,
)


# -----------------------------------------------------------------------
# bh_adjust
# -----------------------------------------------------------------------

class TestBHAdjust:
    def test_known_values(self):
        p = [0.01, 0.04, 0.03, 0.005]
        q = bh_adjust(p)
        # All adjusted values should be >= raw values
        assert all(q >= np.array(p))
        # All should be <= 1
        assert all(q <= 1.0)

    def test_single_value(self):
        q = bh_adjust([0.05])
        assert np.isclose(q[0], 0.05)

    def test_all_equal(self):
        p = [0.1, 0.1, 0.1]
        q = bh_adjust(p)
        # Equal p-values should produce equal adjusted values
        assert np.allclose(q, q[0])

    def test_all_zero(self):
        q = bh_adjust([0.0, 0.0, 0.0])
        assert all(q == 0.0)

    def test_all_one(self):
        q = bh_adjust([1.0, 1.0, 1.0])
        assert all(q == 1.0)

    def test_ordering_preserved(self):
        """Smaller raw p-values should still have smaller adjusted values."""
        p = [0.001, 0.01, 0.05, 0.1]
        q = bh_adjust(p)
        assert all(q[i] <= q[i + 1] for i in range(len(q) - 1))


# -----------------------------------------------------------------------
# bonferroni_adjust
# -----------------------------------------------------------------------

class TestBonferroniAdjust:
    def test_known_values(self):
        p = [0.01, 0.04, 0.03]
        q = bonferroni_adjust(p)
        expected = np.minimum(np.array(p) * 3, 1.0)
        np.testing.assert_allclose(q, expected)

    def test_capping_at_one(self):
        p = [0.5, 0.6, 0.8]
        q = bonferroni_adjust(p)
        assert all(q <= 1.0)

    def test_single_value(self):
        q = bonferroni_adjust([0.05])
        assert np.isclose(q[0], 0.05)


# -----------------------------------------------------------------------
# holm_adjust
# -----------------------------------------------------------------------

class TestHolmAdjust:
    def test_known_values(self):
        p = [0.01, 0.04, 0.03, 0.005]
        q = holm_adjust(p)
        assert all(q >= np.array(p))
        assert all(q <= 1.0)

    def test_monotonicity(self):
        """Adjusted p-values should be monotonically non-decreasing when
        sorted by the original p-values."""
        p = [0.001, 0.01, 0.05, 0.1]
        q = holm_adjust(p)
        order = np.argsort(p)
        sorted_q = q[order]
        assert all(sorted_q[i] <= sorted_q[i + 1] for i in range(len(sorted_q) - 1))

    def test_single_value(self):
        q = holm_adjust([0.05])
        assert np.isclose(q[0], 0.05)

    def test_tied_pvalues(self):
        p = [0.05, 0.05, 0.05]
        q = holm_adjust(p)
        assert all(q <= 1.0)
        assert all(q >= 0.05)


# -----------------------------------------------------------------------
# adjust_p_values (dispatcher)
# -----------------------------------------------------------------------

class TestAdjustPValues:
    @pytest.mark.parametrize("method", ["bh", "bonferroni", "holm"])
    def test_dispatch(self, method):
        p = [0.01, 0.04, 0.03]
        q = adjust_p_values(p, method=method)
        assert len(q) == 3
        assert all(q <= 1.0)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown correction method"):
            adjust_p_values([0.05], method="invalid")

    def test_bh_matches_direct(self):
        p = [0.01, 0.05, 0.1]
        np.testing.assert_array_equal(adjust_p_values(p, "bh"), bh_adjust(p))

    def test_bonferroni_matches_direct(self):
        p = [0.01, 0.05, 0.1]
        np.testing.assert_array_equal(
            adjust_p_values(p, "bonferroni"), bonferroni_adjust(p)
        )

    def test_holm_matches_direct(self):
        p = [0.01, 0.05, 0.1]
        np.testing.assert_array_equal(
            adjust_p_values(p, "holm"), holm_adjust(p)
        )
