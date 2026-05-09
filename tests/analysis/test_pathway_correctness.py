"""Pathway enrichment analysis correctness tests.

Tests hypergeometric, Fisher's exact, ORA, and GSEA with known exact values.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom, fisher_exact

from pipeGEM.analysis.pathway._base import (
    hypergeometric_test,
    fishers_exact_test,
    over_representation_analysis,
    gsea_style_enrichment,
)


# =====================================================================
# Hypergeometric exact values
# =====================================================================

class TestHypergeometricExact:

    def test_hypergeometric_exact_pvalue(self):
        """10 rxns, 3 sig total, 5 in pathway, 3 sig in pathway.
        Expected p from scipy: hypergeom.sf(2, 10, 3, 5)."""
        data = pd.DataFrame({
            "pathway": ["P1"] * 5 + ["P2"] * 5,
            "sig": [True, True, True, False, False,
                    False, False, False, False, False],
        })
        result = hypergeometric_test(data, pathway_col="pathway", sig_col="sig")
        expected_p = hypergeom.sf(2, 10, 3, 5)  # sf(k-1, N, K, n)
        assert np.isclose(result.loc["P1", "pval"], expected_p, atol=1e-10)

    def test_hypergeometric_all_sig_in_pathway(self):
        """All significant in one pathway -> very small p."""
        data = pd.DataFrame({
            "pathway": ["P1"] * 3 + ["P2"] * 7,
            "sig": [True, True, True] + [False] * 7,
        })
        result = hypergeometric_test(data, pathway_col="pathway", sig_col="sig")
        assert result.loc["P1", "pval"] < 0.05

    def test_hypergeometric_no_sig_in_pathway(self):
        """Zero significant in pathway -> p = 1.0."""
        data = pd.DataFrame({
            "pathway": ["P1"] * 5 + ["P2"] * 5,
            "sig": [False] * 5 + [True, True, True, False, False],
        })
        result = hypergeometric_test(data, pathway_col="pathway", sig_col="sig")
        assert result.loc["P1", "pval"] == 1.0


# =====================================================================
# Fisher's exact values
# =====================================================================

class TestFishersExact:

    def test_fishers_matches_scipy(self):
        """Build contingency table, compare p-value against scipy."""
        # 10 items, pathway P1 has 4, 3 sig total, 2 sig in P1
        data = pd.DataFrame({
            "pathway": ["P1"] * 4 + ["P2"] * 6,
            "sig": [True, True, False, False,
                    True, False, False, False, False, False],
        })
        result = fishers_exact_test(data, pathway_col="pathway", sig_col="sig")

        # Manual contingency table for P1:
        # a=2 (sig & in P1), b=1 (sig & not in P1), c=2 (not sig & in P1), d=5 (not sig & not in P1)
        _, expected_p = fisher_exact([[2, 1], [2, 5]], alternative="greater")
        assert np.isclose(result.loc["P1", "pval"], expected_p, atol=1e-10)

    def test_fishers_odds_ratio_correct(self):
        """Verify odds ratio matches (a*d)/(b*c) hand calculation."""
        data = pd.DataFrame({
            "pathway": ["P1"] * 4 + ["P2"] * 6,
            "sig": [True, True, True, False,
                    False, False, False, False, False, False],
        })
        result = fishers_exact_test(data, pathway_col="pathway", sig_col="sig")
        # a=3, b=0, c=1, d=6 -> odds_ratio = inf (since b=0)
        or_val, _ = fisher_exact([[3, 0], [1, 6]], alternative="greater")
        assert np.isclose(result.loc["P1", "odds_ratio"], or_val)


# =====================================================================
# ORA exact values
# =====================================================================

class TestORAExact:

    def test_ora_enriched_lower_pval(self):
        """Two pathways, one fully enriched -> enriched has strictly lower p."""
        sig_ids = ["r1", "r2", "r3"]
        pathways = {
            "enriched": ["r1", "r2", "r3", "r4"],
            "background": ["r5", "r6", "r7", "r8"],
        }
        bg = ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"]
        result = over_representation_analysis(sig_ids, pathways, background_ids=bg)
        assert result.loc["enriched", "pval"] < result.loc["background", "pval"]

    def test_ora_background_size_affects_pval(self):
        """Same sig_ids, different background -> different p-values."""
        sig_ids = ["r1", "r2"]
        pathways = {"P1": ["r1", "r2", "r3", "r4"]}

        # Small background
        bg_small = ["r1", "r2", "r3", "r4", "r5"]
        result_small = over_representation_analysis(sig_ids, pathways,
                                                     background_ids=bg_small)
        # Large background
        bg_large = ["r1", "r2", "r3", "r4"] + [f"r{i}" for i in range(5, 50)]
        result_large = over_representation_analysis(sig_ids, pathways,
                                                     background_ids=bg_large)
        assert not np.isclose(result_small.loc["P1", "pval"],
                              result_large.loc["P1", "pval"])


# =====================================================================
# GSEA deterministic tests
# =====================================================================

class TestGSEADeterministic:

    def test_gsea_top_scores_positive_es(self):
        """Pathway members all have the highest absolute scores -> ES > 0."""
        scores = {f"r{i}": 10.0 - i * 0.1 for i in range(20)}
        # Top-scoring reactions in one pathway
        pathways = {"top_pathway": ["r0", "r1", "r2", "r3"]}
        result = gsea_style_enrichment(scores, pathways, n_permutations=500,
                                        seed=42)
        assert result.loc["top_pathway", "ES"] > 0

    def test_gsea_bottom_scores_negative_or_zero_es(self):
        """Pathway members all have the lowest scores -> ES <= 0."""
        scores = {f"r{i}": 10.0 - i * 0.5 for i in range(20)}
        # Bottom-scoring reactions
        pathways = {"bottom_pathway": ["r16", "r17", "r18", "r19"]}
        result = gsea_style_enrichment(scores, pathways, n_permutations=500,
                                        seed=42)
        assert result.loc["bottom_pathway", "ES"] <= 0

    def test_gsea_padj_computed(self):
        """When multiple pathways tested, padj column is present and <= 1."""
        scores = {f"r{i}": float(i) for i in range(30)}
        pathways = {
            "P1": ["r0", "r1", "r2", "r3"],
            "P2": ["r26", "r27", "r28", "r29"],
        }
        result = gsea_style_enrichment(scores, pathways, n_permutations=200,
                                        seed=42)
        assert "padj" in result.columns
        assert (result["padj"] <= 1.0).all()
        assert (result["padj"] >= 0.0).all()
