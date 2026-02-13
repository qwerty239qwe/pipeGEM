"""Tests for pipeGEM/data/preprocessing.py."""
from unittest.mock import patch, MagicMock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeGEM.data.preprocessing import (
    get_gene_id_map,
    translate_gene_id,
    transform_HPA_data,
    unify_score_column,
    CORDA_THRESHOLDS,
    HPA_SCORE_COLS,
)


# =====================================================================
# get_gene_id_map
# =====================================================================

class TestGetGeneIdMap:
    @patch("pipeGEM.data.preprocessing.biomart_query")
    def test_df_path_none_no_caching(self, mock_bq):
        mock_result = MagicMock()
        mock_result.as_dataframe.return_value = pd.DataFrame({
            "ensembl_gene_id": ["ENSG001", "ENSG002"],
            "external_gene_name": ["GeneA", "GeneB"],
        })
        mock_bq.return_value = mock_result

        result = get_gene_id_map(
            gene_names=["ENSG001", "ENSG002"],
            from_id="ensembl_gene_id",
            to_id="external_gene_name",
            df_path=None,
        )
        assert "map_df" in result
        assert isinstance(result["map_df"], pd.DataFrame)
        mock_bq.assert_called_once()

    @patch("pipeGEM.data.preprocessing.biomart_query")
    def test_df_path_not_exists_queries_and_saves(self, mock_bq, tmp_path):
        mock_result = MagicMock()
        mock_result.as_dataframe.return_value = pd.DataFrame({
            "ensembl_gene_id": ["ENSG001"],
            "external_gene_name": ["GeneA"],
        })
        mock_bq.return_value = mock_result

        cache_path = tmp_path / "gene_map.tsv"
        result = get_gene_id_map(
            gene_names=["ENSG001"],
            from_id="ensembl_gene_id",
            to_id="external_gene_name",
            df_path=cache_path,
        )
        assert cache_path.is_file()
        assert "map_df" in result

    @patch("pipeGEM.data.preprocessing.biomart_query")
    def test_df_path_exists_reads_cache(self, mock_bq, tmp_path):
        cache_path = tmp_path / "gene_map.tsv"
        # Pre-create a cache file
        pd.DataFrame({
            "ensembl_gene_id": ["ENSG001"],
            "external_gene_name": ["GeneA"],
        }).to_csv(cache_path, sep="\t")

        result = get_gene_id_map(
            gene_names=["ENSG001"],
            from_id="ensembl_gene_id",
            to_id="external_gene_name",
            df_path=cache_path,
        )
        assert "map_df" in result
        mock_bq.assert_not_called()

    @patch("pipeGEM.data.preprocessing.biomart_query")
    def test_map_type_dict(self, mock_bq):
        mock_result = MagicMock()
        mock_result.as_dataframe.return_value = pd.DataFrame({
            "ensembl_gene_id": ["ENSG001", "ENSG002"],
            "external_gene_name": ["GeneA", "GeneB"],
        })
        mock_bq.return_value = mock_result

        result = get_gene_id_map(
            gene_names=["ENSG001", "ENSG002"],
            from_id="ensembl_gene_id",
            to_id="external_gene_name",
            df_path=None,
            map_type="dict",
        )
        assert isinstance(result["map_df"], dict)
        assert result["map_df"]["ENSG001"] == "GeneA"

    @patch("pipeGEM.data.preprocessing.biomart_query")
    def test_drop_unused_filters_by_model(self, mock_bq):
        mock_result = MagicMock()
        mock_result.as_dataframe.return_value = pd.DataFrame({
            "ensembl_gene_id": ["ENSG001", "ENSG002", "ENSG003"],
            "external_gene_name": ["GeneA", "GeneB", "GeneC"],
        })
        mock_bq.return_value = mock_result

        model = MagicMock()
        gene1 = MagicMock()
        gene1.id = "GeneA"
        model.genes = [gene1]

        result = get_gene_id_map(
            gene_names=["ENSG001", "ENSG002", "ENSG003"],
            from_id="ensembl_gene_id",
            to_id="external_gene_name",
            df_path=None,
            drop_unused=True,
            ref_model=model,
        )
        assert len(result["map_df"]) == 1

    def test_drop_unused_true_without_model_raises(self):
        with pytest.raises(AssertionError):
            get_gene_id_map(
                gene_names=["x"],
                from_id="a",
                to_id="b",
                df_path=None,
                drop_unused=True,
                ref_model=None,
            )


# =====================================================================
# translate_gene_id
# =====================================================================

class TestTranslateGeneId:
    def test_normal_column_mapping(self):
        data_df = pd.DataFrame({
            "gene": ["ENSG001", "ENSG002", "ENSG003"],
            "value": [1.0, 2.0, 3.0],
        })
        map_df = {"ENSG001": "GeneA", "ENSG002": "GeneB"}
        result = translate_gene_id(data_df, map_df, gene_col="gene", to_id="mapped")
        assert "mapped" in result["data_df"].columns
        # ENSG003 not mapped -> should be filtered
        assert len(result["data_df"]) == 2

    def test_gene_col_index(self):
        data_df = pd.DataFrame(
            {"value": [1.0, 2.0]},
            index=["ENSG001", "ENSG002"],
        )
        map_df = {"ENSG001": "GeneA", "ENSG002": "GeneB"}
        result = translate_gene_id(data_df, map_df, gene_col="index", to_id="mapped")
        assert "mapped" in result["data_df"].columns

    def test_to_id_index_sets_index(self):
        data_df = pd.DataFrame({
            "gene": ["ENSG001", "ENSG002"],
            "value": [1.0, 2.0],
        })
        map_df = {"ENSG001": "GeneA", "ENSG002": "GeneB"}
        result = translate_gene_id(data_df, map_df, gene_col="gene", to_id="index")
        assert "GeneA" in result["data_df"].index


# =====================================================================
# transform_HPA_data
# =====================================================================

class TestTransformHPAData:
    def test_single_category(self):
        data_df = pd.DataFrame({
            "entrezgene": ["g1", "g1", "g2", "g2"],
            "score": [1.0, 2.0, 3.0, 4.0],
            "tissue": ["brain", "liver", "brain", "liver"],
        })
        result = transform_HPA_data(data_df, categories=["tissue"])
        df = result["data_df"]
        assert "brain" in df.columns
        assert "liver" in df.columns

    def test_multiple_categories(self):
        data_df = pd.DataFrame({
            "entrezgene": ["g1", "g1", "g2", "g2"],
            "score": [1.0, 2.0, 3.0, 4.0],
            "tissue": ["brain", "liver", "brain", "liver"],
            "cell": ["neuron", "hepatocyte", "neuron", "hepatocyte"],
        })
        result = transform_HPA_data(data_df, categories=["tissue", "cell"])
        df = result["data_df"]
        assert not df.empty

    def test_zero_categories_raises(self):
        data_df = pd.DataFrame({
            "entrezgene": ["g1"],
            "score": [1.0],
        })
        with pytest.raises(ValueError, match="at least one sample col"):
            transform_HPA_data(data_df, categories=[])


# =====================================================================
# unify_score_column
# =====================================================================

class TestUnifyScoreColumn:

    def test_continuous_levels(self):
        """Continuous: level_dic={'High': 2, 'Medium': 1}, compute weighted score."""
        data_df = pd.DataFrame({
            "gene": ["g1", "g2"],
            "High": [6, 0],
            "Medium": [4, 10],
        })
        level_dic = {"High": 2, "Medium": 1}
        result = unify_score_column(data_df, level_dic, score_col_name="score")
        rdf = result["data_df"]
        # g1: (6*2 + 4*1) / (6+4) = 16/10 = 1.6
        assert np.isclose(rdf.loc[0, "score"], 1.6)
        # g2: (0*2 + 10*1) / (0+10) = 10/10 = 1.0
        assert np.isclose(rdf.loc[1, "score"], 1.0)

    def test_continuous_uses_continuous_thresholds(self):
        """Continuous branch -> used_rxn_thres = CORDA_THRESHOLDS['continuous']."""
        data_df = pd.DataFrame({"High": [1], "Medium": [2]})
        result = unify_score_column(data_df, {"High": 2, "Medium": 1}, "score")
        assert result["used_rxn_thres"] == CORDA_THRESHOLDS["continuous"]

    def test_discrete_level_column(self):
        """Discrete: 'Level' col -> values replaced by level_dic mapping."""
        data_df = pd.DataFrame({
            "gene": ["g1", "g2", "g3"],
            "Level": ["HC", "MC", "NC"],
        })
        level_dic = {"HC": 3, "MC": 1, "NC": 0}
        result = unify_score_column(data_df, level_dic, score_col_name="score")
        rdf = result["data_df"]
        assert "score" in rdf.columns
        assert rdf.loc[0, "score"] == 3
        assert rdf.loc[1, "score"] == 1
        assert rdf.loc[2, "score"] == 0
        assert result["used_rxn_thres"] == CORDA_THRESHOLDS["discrete"]

    def test_hpa_column_renamed(self):
        """HPA: df has 'pTPM' column, empty level_dic -> renamed to score_col_name."""
        data_df = pd.DataFrame({"pTPM": [1.5, 2.5, 3.5]})
        result = unify_score_column(data_df, {}, score_col_name="my_score")
        rdf = result["data_df"]
        assert "my_score" in rdf.columns
        assert "pTPM" not in rdf.columns
        assert result["used_rxn_thres"] is None

    def test_level_columns_dropped(self):
        """Level columns should be dropped after score computation."""
        data_df = pd.DataFrame({"High": [5], "Medium": [3], "gene": ["g1"]})
        result = unify_score_column(data_df, {"High": 2, "Medium": 1}, "score")
        rdf = result["data_df"]
        assert "High" not in rdf.columns
        assert "Medium" not in rdf.columns
        assert "score" in rdf.columns

    def test_nx_column_used_as_hpa(self):
        """NX column (second HPA score col) used when pTPM absent."""
        data_df = pd.DataFrame({"NX": [10.0, 20.0]})
        result = unify_score_column(data_df, {}, score_col_name="my_score")
        rdf = result["data_df"]
        assert "my_score" in rdf.columns
