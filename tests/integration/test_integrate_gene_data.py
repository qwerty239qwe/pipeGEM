"""Tests for rFASTCORMICS and INIT integration algorithms,
exercised through both the direct apply_* API and the high-level
``pmod.integrate_gene_data`` interface.
"""
import numpy as np
import pytest
import cobra

from pipeGEM import Model
from pipeGEM.data import GeneData
from pipeGEM.integration.algo.rFASTCORMICS import apply_rFASTCORMICS
from pipeGEM.analysis.results.integration import rFASTCORMICSAnalysis


# ─────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def gene_data(ecoli_core, ecoli_core_data):
    data = GeneData(
        data=ecoli_core_data["sample_0"],
        data_transform=lambda x: np.log2(x),
        absent_expression=-np.inf,
    )
    data.align(ecoli_core)
    return data


@pytest.fixture(scope="module")
def percentile_thres(gene_data):
    return gene_data.get_threshold("percentile", p=[75, 25])


@pytest.fixture(scope="module")
def rfastcormics_thres(gene_data):
    return gene_data.get_threshold("rFASTCORMICS")


# ─────────────────────────────────────────────
# rFASTCORMICS – direct apply_rFASTCORMICS API
# ─────────────────────────────────────────────

class TestRFASTCORMICSDirect:

    def test_returns_rfastcormics_analysis(self, ecoli_core, gene_data, rfastcormics_thres):
        result = apply_rFASTCORMICS(
            model=ecoli_core,
            data=gene_data,
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
            predefined_threshold=rfastcormics_thres,
            threshold_kws={},
        )
        assert isinstance(result, rFASTCORMICSAnalysis)

    def test_result_model_is_cobra_model(self, ecoli_core, gene_data, rfastcormics_thres):
        result = apply_rFASTCORMICS(
            model=ecoli_core,
            data=gene_data,
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
            predefined_threshold=rfastcormics_thres,
            threshold_kws={},
        )
        assert isinstance(result.result_model, cobra.Model)

    def test_result_model_reactions_subset_of_original(self, ecoli_core, gene_data,
                                                        rfastcormics_thres):
        original_ids = {r.id for r in ecoli_core.reactions}
        result = apply_rFASTCORMICS(
            model=ecoli_core,
            data=gene_data,
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
            predefined_threshold=rfastcormics_thres,
            threshold_kws={},
        )
        kept_ids = {r.id for r in result.result_model.reactions}
        assert kept_ids <= original_ids

    def test_protected_rxn_always_present(self, ecoli_core, gene_data, rfastcormics_thres):
        protected = ["BIOMASS_Ecoli_core_w_GAM"]
        result = apply_rFASTCORMICS(
            model=ecoli_core,
            data=gene_data,
            protected_rxns=protected,
            predefined_threshold=rfastcormics_thres,
            threshold_kws={},
        )
        kept_ids = {r.id for r in result.result_model.reactions}
        for p in protected:
            assert p in kept_ids

    def test_result_model_is_feasible(self, ecoli_core, gene_data, rfastcormics_thres):
        result = apply_rFASTCORMICS(
            model=ecoli_core,
            data=gene_data,
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
            predefined_threshold=rfastcormics_thres,
            threshold_kws={},
        )
        sol = result.result_model.optimize()
        assert sol.status == "optimal"

    def test_core_rxns_attribute_populated(self, ecoli_core, gene_data, rfastcormics_thres):
        result = apply_rFASTCORMICS(
            model=ecoli_core,
            data=gene_data,
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
            predefined_threshold=rfastcormics_thres,
            threshold_kws={},
        )
        assert isinstance(result.core_rxns, (set, frozenset))
        assert len(result.core_rxns) > 0

    def test_removed_rxn_ids_are_numpy_array(self, ecoli_core, gene_data, rfastcormics_thres):
        result = apply_rFASTCORMICS(
            model=ecoli_core,
            data=gene_data,
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
            predefined_threshold=rfastcormics_thres,
            threshold_kws={},
        )
        assert isinstance(result.removed_rxn_ids, np.ndarray)

    def test_threshold_analysis_stored(self, ecoli_core, gene_data, rfastcormics_thres):
        result = apply_rFASTCORMICS(
            model=ecoli_core,
            data=gene_data,
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
            predefined_threshold=rfastcormics_thres,
            threshold_kws={},
        )
        assert result.threshold_analysis is not None

    def test_twostep_method(self, ecoli_core, gene_data, rfastcormics_thres):
        result = apply_rFASTCORMICS(
            model=ecoli_core,
            data=gene_data,
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
            predefined_threshold=rfastcormics_thres,
            threshold_kws={},
            method="twostep",
        )
        assert isinstance(result, rFASTCORMICSAnalysis)
        assert isinstance(result.result_model, cobra.Model)

    def test_twostep_result_model_subset_of_original(self, ecoli_core, gene_data,
                                                       rfastcormics_thres):
        original_ids = {r.id for r in ecoli_core.reactions}
        result = apply_rFASTCORMICS(
            model=ecoli_core,
            data=gene_data,
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
            predefined_threshold=rfastcormics_thres,
            threshold_kws={},
            method="twostep",
        )
        kept_ids = {r.id for r in result.result_model.reactions}
        assert kept_ids <= original_ids

    def test_original_model_not_mutated(self, ecoli_core, gene_data, rfastcormics_thres):
        n_before = len(ecoli_core.reactions)
        apply_rFASTCORMICS(
            model=ecoli_core,
            data=gene_data,
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
            predefined_threshold=rfastcormics_thres,
            threshold_kws={},
        )
        assert len(ecoli_core.reactions) == n_before

    def test_algo_efficacy_is_float(self, ecoli_core, gene_data, rfastcormics_thres):
        result = apply_rFASTCORMICS(
            model=ecoli_core,
            data=gene_data,
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
            predefined_threshold=rfastcormics_thres,
            threshold_kws={},
        )
        assert isinstance(result.algo_efficacy, float)
        assert 0.0 <= result.algo_efficacy <= 1.0


# ─────────────────────────────────────────────
# rFASTCORMICS – high-level pmod interface
# ─────────────────────────────────────────────

class TestRFASTCORMICSHighLevel:

    def test_via_integrate_gene_data(self, ecoli_core, gene_data, rfastcormics_thres):
        pmod = Model(model=ecoli_core, name_tag="ecoli")
        pmod.add_gene_data("s0", gene_data)
        result = pmod.integrate_gene_data(
            data_name="s0",
            integrator="rFASTCORMICS",
            predefined_threshold=rfastcormics_thres,
            threshold_kws={},
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
        )
        assert isinstance(result, rFASTCORMICSAnalysis)

    def test_result_model_smaller_or_equal(self, ecoli_core, gene_data, rfastcormics_thres):
        pmod = Model(model=ecoli_core, name_tag="ecoli")
        pmod.add_gene_data("s0", gene_data)
        result = pmod.integrate_gene_data(
            data_name="s0",
            integrator="rFASTCORMICS",
            predefined_threshold=rfastcormics_thres,
            threshold_kws={},
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
        )
        assert len(result.result_model.reactions) <= len(ecoli_core.reactions)

    def test_percentile_threshold_works(self, ecoli_core, gene_data, percentile_thres):
        pmod = Model(model=ecoli_core, name_tag="ecoli")
        pmod.add_gene_data("s0", gene_data)
        result = pmod.integrate_gene_data(
            data_name="s0",
            integrator="rFASTCORMICS",
            predefined_threshold=percentile_thres,
            threshold_kws={},
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
        )
        assert isinstance(result.result_model, cobra.Model)


# ─────────────────────────────────────────────
# Cross-checks: rFASTCORMICS vs ecoli_core invariants
# ─────────────────────────────────────────────

class TestRFASTCORMICSInvariants:

    @pytest.fixture(scope="class")
    def rfc_result(self, ecoli_core, gene_data, rfastcormics_thres):
        return apply_rFASTCORMICS(
            model=ecoli_core,
            data=gene_data,
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
            predefined_threshold=rfastcormics_thres,
            threshold_kws={},
        )

    def test_core_rxns_subset_of_original(self, rfc_result, ecoli_core):
        original_ids = {r.id for r in ecoli_core.reactions}
        assert rfc_result.core_rxns <= original_ids

    def test_kept_rxn_ids_match_result_model(self, rfc_result):
        kept_in_model = {r.id for r in rfc_result.result_model.reactions}
        kept_attr = set(rfc_result.kept_rxn_ids)
        assert kept_in_model == kept_attr

    def test_removed_and_kept_partition_original(self, rfc_result, ecoli_core):
        original_ids = {r.id for r in ecoli_core.reactions}
        kept_ids = set(rfc_result.kept_rxn_ids)
        removed_ids = set(rfc_result.removed_rxn_ids)
        assert kept_ids | removed_ids <= original_ids
        assert len(kept_ids & removed_ids) == 0

    def test_biomass_in_kept_reactions(self, rfc_result):
        assert "BIOMASS_Ecoli_core_w_GAM" in set(rfc_result.kept_rxn_ids)
