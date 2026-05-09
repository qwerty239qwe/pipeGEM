"""Tests for pipeGEM/analysis/_stat.py — statistical tester classes."""
import numpy as np
import pandas as pd
import pytest

from pipeGEM.analysis._stat import (
    NormalityTester,
    HomoscedasticityTester,
    PairwiseTester,
    MultiGroupComparison,
)


# =====================================================================
# Helpers
# =====================================================================

def _make_data(groups, dv="value", between="group", seed=42):
    """Create a long-format DataFrame from a dict of {group_name: array}."""
    rng = np.random.RandomState(seed)
    rows = []
    for name, vals in groups.items():
        for v in vals:
            rows.append({dv: v, between: name})
    return pd.DataFrame(rows)


# =====================================================================
# NormalityTester
# =====================================================================

class TestNormalityTester:

    def test_normal_data_passes(self):
        """Normal data (n=50) -> normal=True."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({"value": rng.normal(0, 1, 50)})
        result = NormalityTester.test(data, dv="value")
        assert result.result_df["normal"].iloc[0] == True

    def test_uniform_data_fails(self):
        """Uniform data -> normal=False."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({"value": rng.uniform(0, 1, 50)})
        result = NormalityTester.test(data, dv="value")
        # Uniform may or may not pass depending on sample; use larger sample
        data_large = pd.DataFrame({"value": rng.uniform(0, 1, 500)})
        result_large = NormalityTester.test(data_large, dv="value",
                                             method="normaltest")
        assert result_large.result_df["normal"].iloc[0] == False

    def test_grouped_returns_per_group(self):
        """Grouped data returns per-group rows."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
            "value": np.concatenate([rng.normal(0, 1, 30), rng.normal(5, 1, 30)]),
            "group": ["A"] * 30 + ["B"] * 30,
        })
        result = NormalityTester.test(data, dv="value", group="group")
        assert len(result.result_df) == 2
        assert set(result.result_df.index) == {"A", "B"}

    def test_small_sample_skipped(self):
        """Small sample (n=2) -> skipped with warning."""
        data = pd.DataFrame({
            "value": [1.0, 2.0, 3.0, 4.0],
            "group": ["A", "A", "B", "B"],
        })
        result = NormalityTester.test(data, dv="value", group="group")
        # n=2 per group -> warning emitted, p=1 returned
        assert (result.result_df["p-value"] == 1.0).all()


# =====================================================================
# HomoscedasticityTester
# =====================================================================

class TestHomoscedasticityTester:

    def test_equal_variance_passes(self):
        """Equal variance groups -> equal_var=True."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
            "value": np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)]),
            "group": ["A"] * 50 + ["B"] * 50,
        })
        result = HomoscedasticityTester.test(data, dv="value", group="group")
        assert result.result_df["equal_var"].iloc[0] == True

    def test_very_different_variances_fails(self):
        """Very different variances -> equal_var=False."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
            "value": np.concatenate([rng.normal(0, 0.01, 50), rng.normal(0, 100, 50)]),
            "group": ["A"] * 50 + ["B"] * 50,
        })
        result = HomoscedasticityTester.test(data, dv="value", group="group")
        assert result.result_df["equal_var"].iloc[0] == False

    def test_single_group_returns_default(self):
        """Single group -> returns (stat=0, p=1)."""
        data = pd.DataFrame({
            "value": [1.0, 2.0, 3.0],
            "group": ["A", "A", "A"],
        })
        result = HomoscedasticityTester.test(data, dv="value", group="group")
        assert result.result_df["p-value"].iloc[0] == 1.0


# =====================================================================
# PairwiseTester
# =====================================================================

class TestPairwiseTester:

    def test_three_groups_returns_pairwise(self):
        """3 groups -> result_df with pairwise comparisons."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
            "value": np.concatenate([
                rng.normal(0, 1, 20),
                rng.normal(5, 1, 20),
                rng.normal(10, 1, 20),
            ]),
            "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        })
        tester = PairwiseTester()
        result = tester.test(data, dep_var="value", between="group",
                             parametric=False, method="mw")
        assert result.result_df is not None
        assert len(result.result_df) >= 3  # At least 3 pairwise comparisons

    def test_parametric_auto_selects(self):
        """parametric='auto' auto-selects based on assumptions."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
            "value": np.concatenate([
                rng.normal(0, 1, 30),
                rng.normal(5, 1, 30),
            ]),
            "group": ["A"] * 30 + ["B"] * 30,
        })
        tester = PairwiseTester()
        # Use tukey for parametric pool (auto may resolve to True for normal data)
        result = tester.test(data, dep_var="value", between="group",
                             parametric="auto", method="tukey")
        assert result.result_df is not None


# =====================================================================
# MultiGroupComparison
# =====================================================================

class TestMultiGroupComparison:

    def test_parametric_true_anova(self):
        """parametric=True -> ANOVA result."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
            "value": np.concatenate([
                rng.normal(0, 1, 20),
                rng.normal(5, 1, 20),
                rng.normal(10, 1, 20),
            ]),
            "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        })
        comp = MultiGroupComparison()
        result = comp.test(data, dep_var="value", between="group",
                           parametric=True)
        assert result.result_df is not None
        assert "p-unc" in result.result_df.columns

    def test_parametric_false_kruskal(self):
        """parametric=False -> Kruskal result."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
            "value": np.concatenate([
                rng.exponential(1, 20),
                rng.exponential(10, 20),
                rng.exponential(100, 20),
            ]),
            "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        })
        comp = MultiGroupComparison()
        result = comp.test(data, dep_var="value", between="group",
                           parametric=False)
        assert result.result_df is not None

    def test_parametric_auto_infers(self):
        """parametric='auto' -> auto-infers from data."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
            "value": np.concatenate([
                rng.normal(0, 1, 30),
                rng.normal(5, 1, 30),
                rng.normal(10, 1, 30),
            ]),
            "group": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
        })
        comp = MultiGroupComparison()
        result = comp.test(data, dep_var="value", between="group",
                           parametric="auto")
        # The inferred parametric choice is stored
        assert result.result.get("inferred_parametric") in (True, False)
