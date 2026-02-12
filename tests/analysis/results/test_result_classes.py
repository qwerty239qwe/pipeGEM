"""Tests for result classes: GECKOLightAnalysis, GECKOFullAnalysis, AutoStatResult."""
import pandas as pd
import pytest

from pipeGEM.analysis.results.ec import GECKOLightAnalysis, GECKOFullAnalysis
from pipeGEM.analysis.results.auto_stat import AutoStatResult


# =====================================================================
# GECKOLightAnalysis
# =====================================================================

class TestGECKOLightAnalysis:
    def test_property_accessors_with_data(self):
        res = GECKOLightAnalysis(log={"sigma": 0.5})
        res.add_result({
            "modified_bounds": {"rxn1": (0, 100)},
            "kcat_mapping": {"rxn1": 10.0},
            "enzyme_usage": pd.DataFrame({"reaction": ["rxn1"]}),
        })
        assert res.modified_bounds == {"rxn1": (0, 100)}
        assert res.kcat_mapping == {"rxn1": 10.0}
        assert isinstance(res.enzyme_usage, pd.DataFrame)
        assert len(res.enzyme_usage) == 1

    def test_defaults_when_keys_missing(self):
        res = GECKOLightAnalysis(log={})
        assert res.modified_bounds == {}
        assert res.kcat_mapping == {}
        assert isinstance(res.enzyme_usage, pd.DataFrame)
        assert res.enzyme_usage.empty

    def test_add_result_updates(self):
        res = GECKOLightAnalysis(log={})
        res.add_result({"kcat_mapping": {"r1": 1.0}})
        assert res.kcat_mapping == {"r1": 1.0}
        res.add_result({"kcat_mapping": {"r2": 2.0}})
        assert res.kcat_mapping == {"r2": 2.0}

    def test_log_stored(self):
        res = GECKOLightAnalysis(log={"sigma": 0.5, "ptot": 0.3})
        assert res.log["sigma"] == 0.5


# =====================================================================
# GECKOFullAnalysis
# =====================================================================

class TestGECKOFullAnalysis:
    def test_property_accessors_with_data(self):
        res = GECKOFullAnalysis(log={"sigma": 0.5})
        res.add_result({
            "ec_model": "mock_model",
            "protein_pool_id": "prot_pool",
            "draw_reactions": ["draw_p1"],
            "arm_reactions": ["arm_r1"],
        })
        assert res.ec_model == "mock_model"
        assert res.protein_pool_id == "prot_pool"
        assert res.draw_reactions == ["draw_p1"]
        assert res.arm_reactions == ["arm_r1"]

    def test_defaults_when_keys_missing(self):
        res = GECKOFullAnalysis(log={})
        assert res.ec_model is None
        assert res.protein_pool_id == "prot_pool"
        assert res.draw_reactions == []
        assert res.arm_reactions == []


# =====================================================================
# AutoStatResult
# =====================================================================

class TestAutoStatResult:
    @pytest.fixture
    def populated_result(self):
        res = AutoStatResult(log={
            "alpha": 0.05,
            "correction": "bh",
            "n_groups": 2,
            "parametric": True,
        })
        main_df = pd.DataFrame({"T": [5.5], "p-val": [0.001], "test": ["t-test"]})
        res.add_result({
            "main_result": main_df,
            "pairwise_result": None,
            "normality": {
                "method": {"A": "shapiro", "B": "shapiro"},
                "statistic": {"A": 0.98, "B": 0.97},
                "p_value": {"A": 0.5, "B": 0.6},
                "is_normal": {"A": True, "B": True},
            },
            "homoscedasticity": {
                "method": "levene", "statistic": 1.2,
                "p_value": 0.3, "is_equal_var": True,
            },
            "effect_sizes": {"cohens_d": 1.5},
            "sample_sizes": {"A": 20, "B": 20},
            "corrected_p_values": None,
        })
        return res

    def test_main_result(self, populated_result):
        assert isinstance(populated_result.main_result, pd.DataFrame)
        assert "test" in populated_result.main_result.columns

    def test_pairwise_result_none_for_two_groups(self, populated_result):
        assert populated_result.pairwise_result is None

    def test_assumption_results(self, populated_result):
        ar = populated_result.assumption_results
        assert "normality" in ar
        assert "homoscedasticity" in ar
        assert ar["normality"]["is_normal"]["A"] is True

    def test_effect_sizes(self, populated_result):
        assert populated_result.effect_sizes["cohens_d"] == 1.5

    def test_sample_sizes(self, populated_result):
        assert populated_result.sample_sizes["A"] == 20

    def test_summary_text(self, populated_result):
        text = populated_result.summary(format="text")
        assert isinstance(text, str)
        assert "AutoStatTest Summary" in text

    def test_summary_dataframe(self, populated_result):
        df = populated_result.summary(format="dataframe")
        assert isinstance(df, pd.DataFrame)

    def test_summary_latex(self, populated_result):
        latex = populated_result.summary(format="latex")
        assert isinstance(latex, str)
        assert "\\begin" in latex or "tabular" in latex or "test" in latex

    def test_default_empty_result(self):
        res = AutoStatResult(log={})
        assert isinstance(res.main_result, pd.DataFrame)
        assert res.main_result.empty
        assert res.pairwise_result is None
        assert res.effect_sizes == {}
        assert res.sample_sizes == {}
        assert res.corrected_p_values is None
