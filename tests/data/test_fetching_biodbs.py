"""Tests for biodbs-related functions in pipeGEM/data/fetching.py."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from pipeGEM.data.fetching import fetch_HPA_data


class TestFetchHPAData:
    @patch("pipeGEM.data.fetching.hpa_search")
    def test_file_not_exists_calls_hpa_search(self, mock_hpa, tmp_path):
        mock_hpa.return_value = pd.DataFrame({"gene": ["A"], "value": [1]})

        result = fetch_HPA_data("test_data", data_path=tmp_path)
        mock_hpa.assert_called_once_with("test_data")
        assert "data_path" in result
        assert Path(result["data_path"]).exists()

    @patch("pipeGEM.data.fetching.hpa_search")
    def test_file_exists_skips_fetch(self, mock_hpa, tmp_path):
        # Pre-create the file
        tsv_path = (tmp_path / "test_data").with_suffix(".tsv")
        tsv_path.write_text("gene\tvalue\nA\t1\n")

        result = fetch_HPA_data("test_data", data_path=tmp_path)
        mock_hpa.assert_not_called()
        assert result["data_path"] == tsv_path

    @patch("pipeGEM.data.fetching.hpa_search")
    def test_string_path_converted(self, mock_hpa, tmp_path):
        mock_hpa.return_value = pd.DataFrame({"gene": ["A"], "value": [1]})
        result = fetch_HPA_data("test_data", data_path=str(tmp_path))
        assert isinstance(result["data_path"], Path)

    @patch("pipeGEM.data.fetching.hpa_search")
    def test_returned_dict_has_data_path_key(self, mock_hpa, tmp_path):
        mock_hpa.return_value = pd.DataFrame({"gene": ["A"], "value": [1]})
        result = fetch_HPA_data("test_data", data_path=tmp_path)
        assert "data_path" in result

    @patch("pipeGEM.data.fetching.hpa_search")
    def test_saved_file_is_tsv(self, mock_hpa, tmp_path):
        mock_hpa.return_value = pd.DataFrame({"gene": ["A", "B"], "value": [1, 2]})
        result = fetch_HPA_data("my_data", data_path=tmp_path)
        assert str(result["data_path"]).endswith(".tsv")

    @patch("pipeGEM.data.fetching.hpa_search")
    def test_creates_directory_if_needed(self, mock_hpa, tmp_path):
        mock_hpa.return_value = pd.DataFrame({"gene": ["A"], "value": [1]})
        nested = tmp_path / "sub" / "dir"
        result = fetch_HPA_data("test_data", data_path=nested)
        assert nested.exists()
        assert "data_path" in result
