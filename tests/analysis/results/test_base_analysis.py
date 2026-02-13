"""Tests for pipeGEM/analysis/results/_base.py — BaseAnalysis class."""
import pytest

from pipeGEM.analysis.results._base import BaseAnalysis


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
