"""Tests for pipeGEM/analysis/results/_base.py — BaseAnalysis class."""
import pytest
import numpy as np

from pipeGEM.analysis.results._base import BaseAnalysis
from pipeGEM.data import GeneData


class TestBaseAnalysis:

    @pytest.fixture
    def analysis(self):
        ba = BaseAnalysis(log={"method": "test", "alpha": 0.05})
        ba.add_result({"flux_df": [1, 2, 3], "model": "test_model"})
        return ba

    def test_format_str_includes_class_name(self, analysis):
        s = analysis.format_str()
        assert "BaseAnalysis" in s

    def test_format_str_includes_log_keys(self, analysis):
        s = analysis.format_str()
        assert "method" in s
        assert "alpha" in s

    def test_format_str_includes_result_keys(self, analysis):
        s = analysis.format_str()
        assert "flux_df" in s
        assert "model" in s

    def test_repr_equals_str(self, analysis):
        assert repr(analysis) == str(analysis)

    def test_str_equals_format_str(self, analysis):
        assert str(analysis) == analysis.format_str()

    def test_getattr_falls_back_to_result(self, analysis):
        """__getattr__ falls back to _result dict for known keys."""
        assert analysis.flux_df == [1, 2, 3]
        assert analysis.model == "test_model"

    def test_getattr_raises_for_unknown(self, analysis):
        """__getattr__ raises AttributeError for unknown keys."""
        with pytest.raises(AttributeError):
            _ = analysis.nonexistent_key

    def test_add_result_merges(self, analysis):
        analysis.add_result({"extra_key": 42})
        assert analysis.result["extra_key"] == 42
        # Original keys still present
        assert "flux_df" in analysis.result

    def test_schema_validation_rejects_unknown_when_strict(self):
        class StrictAnalysis(BaseAnalysis):
            RESULT_FIELDS = {"known": {"required": True}}
            ALLOW_EXTRA_RESULT_FIELDS = False

        result = StrictAnalysis(log={})
        result.add_result({"known": 1})
        assert result.known == 1
        with pytest.raises(KeyError):
            result.add_result({"unknown": 2})

    def test_validate_result_checks_required_fields(self):
        class StrictAnalysis(BaseAnalysis):
            RESULT_FIELDS = {"known": {"required": True}}

        result = StrictAnalysis(log={})
        with pytest.raises(KeyError):
            result.validate_result()
        result.add_result({"known": 1})
        result.validate_result()

    def test_add_running_time(self, analysis):
        analysis.add_running_time(1.234)
        s = analysis.format_str()
        assert "1.234" in s

    def test_log_property(self, analysis):
        assert analysis.log == {"method": "test", "alpha": 0.05}

    def test_result_property(self, analysis):
        result = analysis.result
        assert isinstance(result, dict)
        assert "flux_df" in result
        assert "model" in result

    def test_running_time_not_shown_when_none(self):
        ba = BaseAnalysis(log={})
        s = ba.format_str()
        assert "Running time" not in s


def test_percentile_threshold_round_trip(tmp_path):
    result = GeneData({"g1": 1., "g2": 4., "g3": 9.}).get_threshold("percentile", p=[25, 75])
    result.save(tmp_path / "threshold")
    loaded = type(result).load(tmp_path / "threshold")
    assert loaded.exp_th == result.exp_th
    assert loaded.non_exp_th == result.non_exp_th
    np.testing.assert_array_equal(loaded.data, result.data)


def test_load_result_selects_only_requested_scalar(tmp_path):
    result = BaseAnalysis(log={})
    result.add_result({"score": 3, "other_score": 7})
    result.save(tmp_path / "analysis")
    loaded = BaseAnalysis.load_result(
        tmp_path / "analysis/result/score", key="score", result_type="python.int",
    )
    assert loaded.result == {"score": 3}
