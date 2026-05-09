"""Tests for pathway analysis modules.

Covers:
- pipeGEM/analysis/pathway/_base.py  (enrichment functions)
- pipeGEM/analysis/pathway/_analysis.py  (PathwayAnalyzer)
- pipeGEM/analysis/pathway/_kegg.py  (KEGGPathwayMapper, mocked HTTP)
"""
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from pipeGEM.analysis.pathway._base import (
    hypergeometric_test,
    fishers_exact_test,
    gsea_style_enrichment,
    over_representation_analysis,
)
from pipeGEM.analysis.pathway._analysis import PathwayAnalyzer
from pipeGEM.analysis.pathway._kegg import KEGGPathwayMapper


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def enrichment_df():
    """DataFrame with pathway and significance columns."""
    return pd.DataFrame({
        "pathway": ["glycolysis"] * 5 + ["tca"] * 5,
        "significant": [True, True, True, False, False,
                        False, False, False, False, True],
    })


@pytest.fixture
def pathway_defs():
    return {
        "glycolysis": ["r1", "r2", "r3", "r4"],
        "tca": ["r5", "r6", "r7"],
        "ppp": ["r8"],
    }


@pytest.fixture
def flux_df():
    return pd.DataFrame(
        {"cond1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]},
        index=["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"],
    )


# =====================================================================
# _base.py — hypergeometric_test
# =====================================================================

class TestHypergeometricTest:
    def test_basic_enrichment(self, enrichment_df):
        result = hypergeometric_test(enrichment_df, "pathway", "significant")
        assert "pval" in result.columns
        assert "padj" in result.columns
        assert "BgRatio" in result.columns
        assert "SigRatio" in result.columns
        assert len(result) == 2  # glycolysis and tca

    def test_no_significance(self):
        df = pd.DataFrame({
            "pathway": ["a", "a", "b", "b"],
            "sig": [False, False, False, False],
        })
        result = hypergeometric_test(df, "pathway", "sig")
        # All p-values should be 1 when nothing is significant
        assert all(result["pval"] == 1.0)

    def test_single_pathway(self):
        df = pd.DataFrame({
            "pw": ["only"] * 4,
            "sig": [True, True, False, False],
        })
        result = hypergeometric_test(df, "pw", "sig")
        assert len(result) == 1

    def test_enriched_pathway_has_lower_pval(self, enrichment_df):
        result = hypergeometric_test(enrichment_df, "pathway", "significant")
        # glycolysis has 3/4 sig (enriched); tca has 1/4 sig
        assert result.loc["glycolysis", "pval"] < result.loc["tca", "pval"]


# =====================================================================
# _base.py — fishers_exact_test
# =====================================================================

class TestFishersExactTest:
    def test_basic_enrichment(self, enrichment_df):
        result = fishers_exact_test(enrichment_df, "pathway", "significant")
        assert "pval" in result.columns
        assert "padj" in result.columns
        assert "odds_ratio" in result.columns
        assert len(result) == 2

    def test_no_significance(self):
        df = pd.DataFrame({
            "pathway": ["a", "a", "b", "b"],
            "sig": [False, False, False, False],
        })
        result = fishers_exact_test(df, "pathway", "sig")
        assert all(result["pval"] >= 0.5)

    def test_odds_ratio_positive(self, enrichment_df):
        result = fishers_exact_test(enrichment_df, "pathway", "significant")
        assert all(result["odds_ratio"] >= 0)


# =====================================================================
# _base.py — gsea_style_enrichment
# =====================================================================

class TestGSEAStyleEnrichment:
    def test_basic_ranking(self, pathway_defs):
        scores = {f"r{i}": float(i) for i in range(1, 9)}
        result = gsea_style_enrichment(scores, pathway_defs, n_permutations=100, seed=42)
        assert "ES" in result.columns
        assert "NES" in result.columns
        assert "pval" in result.columns
        # ppp has only 1 member -> should be skipped
        assert "ppp" not in result.index

    def test_pathway_with_fewer_than_2_members_skipped(self):
        scores = {"r1": 1.0, "r2": 2.0, "r3": 3.0}
        pathway_defs = {"small": ["r1"], "big": ["r1", "r2", "r3"]}
        result = gsea_style_enrichment(scores, pathway_defs, n_permutations=50)
        assert "small" not in result.index

    def test_all_zero_scores(self, pathway_defs):
        scores = {f"r{i}": 0.0 for i in range(1, 9)}
        result = gsea_style_enrichment(scores, pathway_defs, n_permutations=50)
        # With all-zero scores, ES should be 0
        if not result.empty:
            assert all(result["ES"] == 0.0)

    def test_seed_reproducibility(self, pathway_defs):
        scores = {f"r{i}": float(i) * (-1) ** i for i in range(1, 9)}
        r1 = gsea_style_enrichment(scores, pathway_defs, n_permutations=100, seed=123)
        r2 = gsea_style_enrichment(scores, pathway_defs, n_permutations=100, seed=123)
        pd.testing.assert_frame_equal(r1, r2)


# =====================================================================
# _base.py — over_representation_analysis
# =====================================================================

class TestORA:
    def test_basic_ora(self, pathway_defs):
        sig_ids = ["r1", "r2", "r3"]
        bg_ids = [f"r{i}" for i in range(1, 9)]
        result = over_representation_analysis(sig_ids, pathway_defs, bg_ids)
        assert "pval" in result.columns
        assert "padj" in result.columns

    def test_custom_background(self, pathway_defs):
        sig_ids = ["r1", "r5"]
        bg_ids = ["r1", "r2", "r5", "r6", "r7"]
        result = over_representation_analysis(sig_ids, pathway_defs, bg_ids)
        assert len(result) > 0

    def test_no_background_auto_constructed(self, pathway_defs):
        sig_ids = ["r1", "r2"]
        result = over_representation_analysis(sig_ids, pathway_defs, background_ids=None)
        assert len(result) > 0

    def test_empty_pathway(self):
        sig_ids = ["r1"]
        pathway_defs = {"empty": [], "nonempty": ["r1", "r2"]}
        result = over_representation_analysis(sig_ids, pathway_defs, ["r1", "r2"])
        # empty pathway should be skipped (n_pw == 0)
        assert "empty" not in result.index


# =====================================================================
# _analysis.py — PathwayAnalyzer
# =====================================================================

class TestPathwayAnalyzer:
    @pytest.fixture
    def analyzer(self, pathway_defs):
        model = MagicMock()
        return PathwayAnalyzer(model, pathway_defs)

    def test_aggregate_fluxes_mean(self, analyzer, flux_df):
        result = analyzer.aggregate_fluxes(flux_df, method="mean")
        assert "glycolysis" in result.index
        assert "tca" in result.index
        # glycolysis = mean(1,2,3,4) = 2.5
        assert np.isclose(result.loc["glycolysis", "cond1"], 2.5)

    def test_aggregate_fluxes_sum(self, analyzer, flux_df):
        result = analyzer.aggregate_fluxes(flux_df, method="sum")
        # glycolysis = 1+2+3+4 = 10
        assert np.isclose(result.loc["glycolysis", "cond1"], 10.0)

    @pytest.mark.parametrize("method", ["mean", "median", "sum", "max", "min"])
    def test_aggregate_fluxes_methods(self, analyzer, flux_df, method):
        result = analyzer.aggregate_fluxes(flux_df, method=method)
        assert not result.empty

    def test_aggregate_fluxes_empty_pathway(self, flux_df):
        model = MagicMock()
        pa = PathwayAnalyzer(model, {"missing": ["rx_not_exist"]})
        result = pa.aggregate_fluxes(flux_df)
        assert result.empty

    def test_compare_pathways(self, analyzer, flux_df):
        flux_df2 = flux_df * 2
        flux_df2.columns = ["cond2"]
        result = analyzer.compare_pathways({"cond1": flux_df, "cond2": flux_df2})
        assert "cond1" in result.columns
        assert "cond2" in result.columns

    def test_compare_pathways_empty(self, analyzer):
        result = analyzer.compare_pathways({})
        assert result.empty

    def test_identify_bottlenecks(self, analyzer):
        fva_df = pd.DataFrame(
            {"minimum": [0, 0, 0, 0, 0, 0, 0, 0],
             "maximum": [10, 10, 0.1, 10, 10, 10, 10, 10]},
            index=["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"],
        )
        result = analyzer.identify_bottlenecks(fva_df, threshold=0.1)
        # r3 has range=0.1, pathway avg range is ~7.525, so r3 is a bottleneck
        assert len(result) > 0
        assert "r3" in result["reaction"].values

    def test_identify_bottlenecks_no_bottlenecks(self, analyzer):
        fva_df = pd.DataFrame(
            {"minimum": [0] * 8, "maximum": [10] * 8},
            index=[f"r{i}" for i in range(1, 9)],
        )
        result = analyzer.identify_bottlenecks(fva_df, threshold=0.1)
        assert len(result) == 0

    def test_identify_bottlenecks_zero_avg_range(self, analyzer):
        fva_df = pd.DataFrame(
            {"minimum": [5] * 8, "maximum": [5] * 8},
            index=[f"r{i}" for i in range(1, 9)],
        )
        result = analyzer.identify_bottlenecks(fva_df, threshold=0.1)
        # avg_range == 0 -> skip
        assert len(result) == 0


# =====================================================================
# _kegg.py — KEGGPathwayMapper (mocked HTTP)
# =====================================================================

class TestKEGGPathwayMapper:
    @patch("pipeGEM.analysis.pathway._kegg.requests.get")
    def test_fetch_pathways_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = "path:eco00010\tGlycolysis\npath:eco00020\tTCA cycle\n"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        mapper = KEGGPathwayMapper("eco")
        pathways = mapper.fetch_pathways()
        assert len(pathways) == 2
        assert "path:eco00010" in pathways

    @patch("pipeGEM.analysis.pathway._kegg.requests.get")
    def test_fetch_pathways_network_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("Network error")

        mapper = KEGGPathwayMapper("eco")
        pathways = mapper.fetch_pathways()
        assert pathways == {}

    @patch("pipeGEM.analysis.pathway._kegg.requests.get")
    def test_fetch_pathways_empty_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = ""
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        mapper = KEGGPathwayMapper("eco")
        pathways = mapper.fetch_pathways()
        assert len(pathways) == 0

    @patch("pipeGEM.analysis.pathway._kegg.requests.get")
    def test_fetch_pathway_genes_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = "path:eco00010\teco:b0001\npath:eco00010\teco:b0002\n"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        mapper = KEGGPathwayMapper("eco")
        genes = mapper.fetch_pathway_genes("eco00010")
        assert len(genes) == 2
        assert "b0001" in genes

    @patch("pipeGEM.analysis.pathway._kegg.requests.get")
    def test_fetch_pathway_genes_cached(self, mock_get):
        mapper = KEGGPathwayMapper("eco")
        mapper._pathway_cache["eco00010"] = ["b0001", "b0002"]
        genes = mapper.fetch_pathway_genes("eco00010")
        assert genes == ["b0001", "b0002"]
        mock_get.assert_not_called()

    @patch("pipeGEM.analysis.pathway._kegg.requests.get")
    def test_fetch_pathway_genes_network_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("Timeout")

        mapper = KEGGPathwayMapper("eco")
        genes = mapper.fetch_pathway_genes("eco99999")
        assert genes == []

    @patch.object(KEGGPathwayMapper, "fetch_pathway_genes")
    @patch.object(KEGGPathwayMapper, "fetch_pathways")
    def test_map_to_model_with_provided_pathways(self, mock_fetch_pw, mock_fetch_genes):
        mock_fetch_genes.return_value = ["b0001"]
        model = MagicMock()
        gene_mock = MagicMock()
        gene_mock.id = "b0001"
        rxn_mock = MagicMock()
        rxn_mock.id = "PFK"
        rxn_mock.genes = [gene_mock]
        model.reactions = [rxn_mock]

        mapper = KEGGPathwayMapper("eco")
        result = mapper.map_to_model(model, pathways={"eco00010": "Glycolysis"})
        assert "eco00010" in result
        assert "PFK" in result["eco00010"]
        mock_fetch_pw.assert_not_called()

    @patch.object(KEGGPathwayMapper, "fetch_pathway_genes")
    @patch.object(KEGGPathwayMapper, "fetch_pathways")
    def test_map_to_model_no_matching_genes(self, mock_fetch_pw, mock_fetch_genes):
        mock_fetch_genes.return_value = ["gene_not_in_model"]
        model = MagicMock()
        gene_mock = MagicMock()
        gene_mock.id = "b9999"
        rxn_mock = MagicMock()
        rxn_mock.id = "RXN1"
        rxn_mock.genes = [gene_mock]
        model.reactions = [rxn_mock]

        mapper = KEGGPathwayMapper("eco")
        result = mapper.map_to_model(model, pathways={"eco00010": "Glycolysis"})
        assert len(result) == 0
