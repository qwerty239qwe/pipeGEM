"""Correctness tests for AutoStatTest.

Validates exact statistical outcomes using hand-crafted data
where the expected result is known a priori.
"""
import numpy as np
import pandas as pd
import pytest

from pipeGEM.analysis._auto_stat import AutoStatTest, _cohens_d, _eta_squared


class TestCohensD:

    def test_cohens_d_sign_matches_means(self):
        """group_a mean > group_b mean -> d > 0."""
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = _cohens_d(a, b)
        assert d > 0

    def test_cohens_d_zero_for_identical(self):
        """Same data -> d = 0."""
        a = np.array([5.0, 5.0, 5.0, 5.0])
        d = _cohens_d(a, a)
        assert d == 0.0

    def test_cohens_d_negative_when_reversed(self):
        """group_a mean < group_b mean -> d < 0."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 11.0, 12.0])
        d = _cohens_d(a, b)
        assert d < 0


class TestEtaSquared:

    def test_eta_squared_in_range(self):
        """eta-squared is always in [0, 1]."""
        eta = _eta_squared(10.0, 100.0)
        assert 0.0 <= eta <= 1.0

    def test_eta_squared_near_one_for_separated(self):
        """Very different groups -> eta-squared close to 1."""
        # Three very different groups
        groups = {
            "A": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            "B": np.array([100.0, 100.0, 100.0, 100.0, 100.0]),
            "C": np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0]),
        }
        all_vals = np.concatenate(list(groups.values()))
        grand_mean = np.mean(all_vals)
        ss_between = sum(
            len(v) * (np.mean(v) - grand_mean) ** 2 for v in groups.values()
        )
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        eta = _eta_squared(ss_between, ss_total)
        assert eta > 0.99

    def test_eta_squared_zero_for_zero_between(self):
        """ss_between = 0 -> eta = 0."""
        assert _eta_squared(0.0, 100.0) == 0.0


class TestAutoStatTest:

    def test_identical_groups_high_pvalue(self):
        """Two identical groups -> p approx 1.0."""
        rng = np.random.RandomState(42)
        vals = rng.normal(5.0, 1.0, 30)
        data = pd.DataFrame({
            "value": np.concatenate([vals, vals]),
            "group": ["A"] * 30 + ["B"] * 30,
        })
        ast = AutoStatTest(alpha=0.05)
        result = ast.test(data, dv="value", between="group")
        # p-value should be very high (not significant)
        main = result.main_result
        p_col = "p-val" if "p-val" in main.columns else "p-unc"
        assert main[p_col].iloc[0] > 0.5

    def test_separated_groups_low_pvalue(self):
        """Two groups with 10sigma separation -> p < 0.001."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
            "value": np.concatenate([
                rng.normal(0.0, 1.0, 30),
                rng.normal(10.0, 1.0, 30),
            ]),
            "group": ["A"] * 30 + ["B"] * 30,
        })
        ast = AutoStatTest(alpha=0.05)
        result = ast.test(data, dv="value", between="group")
        main = result.main_result
        p_col = "p-val" if "p-val" in main.columns else "p-unc"
        assert main[p_col].iloc[0] < 0.001

    def test_bonferroni_geq_raw(self):
        """All corrected p-values >= raw p-values (3 groups)."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
            "value": np.concatenate([
                rng.normal(0, 1, 20),
                rng.normal(3, 1, 20),
                rng.normal(6, 1, 20),
            ]),
            "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        })
        ast = AutoStatTest(alpha=0.05, correction="bonferroni")
        result = ast.test(data, dv="value", between="group")
        pw = result.pairwise_result
        if pw is not None and "p-unc" in pw.columns and "p-adj" in pw.columns:
            assert (pw["p-adj"] >= pw["p-unc"] - 1e-12).all()

    def test_parametric_inference_correct(self):
        """Normal data -> parametric=True; highly skewed -> parametric=False."""
        rng = np.random.RandomState(42)
        # Normal data
        normal_data = pd.DataFrame({
            "value": np.concatenate([
                rng.normal(0, 1, 50),
                rng.normal(0, 1, 50),
            ]),
            "group": ["A"] * 50 + ["B"] * 50,
        })
        ast = AutoStatTest(alpha=0.05)
        result_normal = ast.test(normal_data, dv="value", between="group")
        assert result_normal.log["parametric"] == True

        # Highly skewed data (exponential)
        skewed_data = pd.DataFrame({
            "value": np.concatenate([
                rng.exponential(0.1, 50),
                rng.exponential(100.0, 50),
            ]),
            "group": ["A"] * 50 + ["B"] * 50,
        })
        result_skewed = ast.test(skewed_data, dv="value", between="group")
        assert result_skewed.log["parametric"] == False

    def test_three_groups_eta_squared(self):
        """Three well-separated groups -> eta_squared close to 1."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
            "value": np.concatenate([
                rng.normal(0, 0.01, 20),
                rng.normal(100, 0.01, 20),
                rng.normal(1000, 0.01, 20),
            ]),
            "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        })
        ast = AutoStatTest(alpha=0.05, effect_size=True)
        result = ast.test(data, dv="value", between="group")
        assert result.effect_sizes["eta_squared"] > 0.99
