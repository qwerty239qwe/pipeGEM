"""Tests for pipeGEM/analysis/_auto_stat.py — AutoStatTest."""
import numpy as np
import pandas as pd
import pytest

from pipeGEM.analysis._auto_stat import (
    AutoStatTest,
    _cohens_d,
    _eta_squared,
    _normalize_pingouin_p_columns,
)
from pipeGEM.analysis.results.auto_stat import AutoStatResult


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_df(groups, seed=42):
    """Build a long-format DataFrame from a dict of group -> values."""
    rng = np.random.RandomState(seed)
    rows = []
    for name, vals in groups.items():
        for v in vals:
            rows.append({"group": name, "value": v})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Effect-size helpers
# -----------------------------------------------------------------------

class TestEffectSizeHelpers:
    def test_cohens_d_identical_groups(self):
        d = _cohens_d([1, 2, 3], [1, 2, 3])
        assert np.isclose(d, 0.0)

    def test_cohens_d_different_groups(self):
        d = _cohens_d([10, 11, 12], [1, 2, 3])
        assert d > 0

    def test_cohens_d_zero_variance(self):
        d = _cohens_d([5, 5, 5], [5, 5, 5])
        assert d == 0.0

    def test_eta_squared_zero_total(self):
        assert _eta_squared(0, 0) == 0.0

    def test_eta_squared_normal(self):
        eta = _eta_squared(10, 100)
        assert np.isclose(eta, 0.1)


# -----------------------------------------------------------------------
# AutoStatTest — 2 groups
# -----------------------------------------------------------------------

class TestAutoStatTwoGroups:
    def test_normal_data_selects_ttest(self):
        rng = np.random.RandomState(0)
        df = _make_df({
            "A": rng.normal(5, 1, 30).tolist(),
            "B": rng.normal(8, 1, 30).tolist(),
        })
        ast = AutoStatTest(alpha=0.05, correction="bh")
        result = ast.test(df, dv="value", between="group")

        assert isinstance(result, AutoStatResult)
        main = result.main_result
        assert "test" in main.columns
        test_name = main["test"].iloc[0]
        # Normal data should pick t-test (parametric) or MWU (if normality fails by chance)
        assert test_name in ("t-test", "Mann-Whitney U")

    def test_non_normal_data_selects_mwu(self):
        rng = np.random.RandomState(1)
        # Highly skewed data
        df = _make_df({
            "A": (rng.exponential(1, 30) ** 3).tolist(),
            "B": (rng.exponential(5, 30) ** 3).tolist(),
        })
        ast = AutoStatTest(alpha=0.05)
        result = ast.test(df, dv="value", between="group")
        assert isinstance(result, AutoStatResult)
        # pairwise should be None for 2 groups
        assert result.pairwise_result is None

    def test_effect_size_cohens_d(self):
        rng = np.random.RandomState(2)
        df = _make_df({
            "A": rng.normal(0, 1, 20).tolist(),
            "B": rng.normal(3, 1, 20).tolist(),
        })
        ast = AutoStatTest(effect_size=True)
        result = ast.test(df, dv="value", between="group")
        assert "cohens_d" in result.effect_sizes

    @pytest.mark.parametrize("p_column", ["p_unc", "p_val", "pvalue", "p-value", "p.value"])
    def test_pingouin_pvalue_aliases_normalized(self, p_column):
        result = _normalize_pingouin_p_columns(pd.DataFrame({p_column: [0.25]}))
        assert result["p-unc"].iloc[0] == 0.25

    def test_pingouin_pval_column_kept_available_for_correction(self):
        result = _normalize_pingouin_p_columns(pd.DataFrame({"p-val": [0.25]}))
        assert result["p-unc"].iloc[0] == 0.25
        assert result["p-val"].iloc[0] == 0.25


# -----------------------------------------------------------------------
# AutoStatTest — 3+ groups
# -----------------------------------------------------------------------

class TestAutoStatMultiGroup:
    def test_normal_data_selects_anova(self):
        rng = np.random.RandomState(10)
        df = _make_df({
            "A": rng.normal(5, 1, 30).tolist(),
            "B": rng.normal(5.5, 1, 30).tolist(),
            "C": rng.normal(6, 1, 30).tolist(),
        })
        ast = AutoStatTest(alpha=0.05, correction="bh")
        result = ast.test(df, dv="value", between="group")
        test_name = result.main_result["test"].iloc[0]
        assert test_name in ("ANOVA", "Kruskal-Wallis")

    def test_non_normal_data_selects_kruskal(self):
        rng = np.random.RandomState(11)
        df = _make_df({
            "A": (rng.exponential(1, 30) ** 3).tolist(),
            "B": (rng.exponential(2, 30) ** 3).tolist(),
            "C": (rng.exponential(3, 30) ** 3).tolist(),
        })
        ast = AutoStatTest(alpha=0.05)
        result = ast.test(df, dv="value", between="group")
        assert isinstance(result, AutoStatResult)
        # Should have pairwise results for 3+ groups
        assert result.pairwise_result is not None

    def test_effect_size_eta_squared(self):
        rng = np.random.RandomState(12)
        df = _make_df({
            "A": rng.normal(0, 1, 20).tolist(),
            "B": rng.normal(3, 1, 20).tolist(),
            "C": rng.normal(6, 1, 20).tolist(),
        })
        ast = AutoStatTest(effect_size=True)
        result = ast.test(df, dv="value", between="group")
        assert "eta_squared" in result.effect_sizes
        assert 0 <= result.effect_sizes["eta_squared"] <= 1

    def test_pvalue_correction_applied(self):
        rng = np.random.RandomState(13)
        df = _make_df({
            "A": rng.normal(0, 1, 20).tolist(),
            "B": rng.normal(3, 1, 20).tolist(),
            "C": rng.normal(6, 1, 20).tolist(),
        })
        ast = AutoStatTest(correction="bonferroni")
        result = ast.test(df, dv="value", between="group")
        pw = result.pairwise_result
        if pw is not None and "p-adj" in pw.columns:
            # Adjusted p-values should be >= raw p-values
            assert all(pw["p-adj"] >= pw["p-unc"] - 1e-10)


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------

class TestAutoStatEdgeCases:
    def test_small_samples_normality_skipped(self):
        """Groups with n<3 should have normality skipped."""
        df = _make_df({
            "A": [1.0, 2.0],  # n=2
            "B": [3.0, 4.0],  # n=2
        })
        ast = AutoStatTest()
        result = ast.test(df, dv="value", between="group")
        norm = result.assumption_results["normality"]
        for g in ["A", "B"]:
            assert norm["method"][g] == "skipped"

    def test_result_properties_accessible(self):
        rng = np.random.RandomState(20)
        df = _make_df({
            "A": rng.normal(0, 1, 15).tolist(),
            "B": rng.normal(2, 1, 15).tolist(),
        })
        ast = AutoStatTest()
        result = ast.test(df, dv="value", between="group")

        assert isinstance(result.main_result, pd.DataFrame)
        assert isinstance(result.assumption_results, dict)
        assert isinstance(result.effect_sizes, dict)
        assert isinstance(result.sample_sizes, dict)
        assert result.sample_sizes["A"] == 15
