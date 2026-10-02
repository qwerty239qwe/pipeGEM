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
from pipeGEM.integration._class import RemovableGeneDataIntegrator


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


# ─────────────────────────────────────────────
# RemovableGeneDataIntegrator.apply / remove round-trip
# ─────────────────────────────────────────────
class _ProbeIntegrator(RemovableGeneDataIntegrator):
    """Concrete integrator that records delegation and mutates a bound."""

    def integrate(self, model, data, **kwargs):
        self._model = model
        self.seen = (model, data, kwargs)
        # mutate inside the context opened by apply(); should roll back on remove()
        model.reactions.get_by_id(data).lower_bound = -123.0
        return "integrated"


class TestRemovableGeneDataIntegratorApply:
    def test_apply_enters_context_and_delegates(self, ecoli_core):
        rxn_id = "PFK"
        original_lb = ecoli_core.reactions.get_by_id(rxn_id).lower_bound
        integ = _ProbeIntegrator()

        result = integ.apply(ecoli_core, rxn_id, foo=1)

        # return value propagated from integrate()
        assert result == "integrated"
        # model + data + kwargs forwarded
        assert integ.seen[0] is ecoli_core
        assert integ.seen[1] == rxn_id
        assert integ.seen[2] == {"foo": 1}
        # _model set, mutation visible while context open
        assert integ._model is ecoli_core
        assert ecoli_core.reactions.get_by_id(rxn_id).lower_bound == -123.0

        # remove() exits context -> cobra rolls the bound change back
        integ.remove()
        assert ecoli_core.reactions.get_by_id(rxn_id).lower_bound == original_lb

    def test_remove_without_apply_is_noop(self):
        # _model is None before apply(); remove() must not raise
        integ = _ProbeIntegrator()
        integ.remove()

    @pytest.mark.parametrize("wrapped", [False, True])
    @pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
    def test_failed_apply_restores_model_and_preserves_outer_context(self, ecoli_core, monkeypatch, wrapped, error_type):
        ecoli_core = ecoli_core.copy()
        model = Model(model=ecoli_core) if wrapped else ecoli_core
        integ = _ProbeIntegrator()
        integrate = integ.integrate
        error = error_type("integration failed")

        def fail(model, data, **kwargs):
            integrate(model, data, **kwargs)
            raise error

        monkeypatch.setattr(integ, "integrate", fail)
        with model:
            model.reactions.PFK.lower_bound = 1.
            with pytest.raises(error_type) as caught:
                integ.apply(model, "PFK")
            assert caught.value is error
            assert model.reactions.PFK.lower_bound == 1.
            assert len(ecoli_core._contexts) == 1
            integ.remove()
            assert len(ecoli_core._contexts) == 1
            monkeypatch.setattr(integ, "integrate", integrate)
            assert integ.apply(model, "PFK") == "integrated"
            integ.remove()
            assert model.reactions.PFK.lower_bound == 1.
        assert len(ecoli_core._contexts) == 0

    @pytest.mark.parametrize("wrapped", [False, True])
    def test_repeated_remove_preserves_outer_context(self, ecoli_core, wrapped):
        ecoli_core = ecoli_core.copy()
        model = Model(model=ecoli_core) if wrapped else ecoli_core
        integ = _ProbeIntegrator()
        with model:
            model.reactions.PFK.lower_bound = 1.
            integ.apply(model, "PFK")
            integ.remove()
            integ.remove()
            assert model.reactions.PFK.lower_bound == 1.
            assert len(ecoli_core._contexts) == 1
        assert len(ecoli_core._contexts) == 0

    def test_repeated_apply_preserves_first_model_context(self, ecoli_core):
        ecoli_core = ecoli_core.copy()
        other_model = ecoli_core.copy()
        integ = _ProbeIntegrator()
        original_lb = ecoli_core.reactions.PFK.lower_bound
        other_lb = other_model.reactions.PFK.lower_bound
        integ.apply(ecoli_core, "PFK")
        try:
            with pytest.raises(RuntimeError, match="already applied"):
                integ.apply(other_model, "PFK")
            assert len(ecoli_core._contexts) == 1
            assert len(other_model._contexts) == 0
            assert other_model.reactions.PFK.lower_bound == other_lb
        finally:
            integ.remove()
        assert ecoli_core.reactions.PFK.lower_bound == original_lb

    def test_remove_after_direct_integrate_preserves_outer_context(self, ecoli_core):
        model = ecoli_core.copy()
        integ = _ProbeIntegrator()
        with model:
            integ.integrate(model, "PFK")
            integ.remove()
            assert model.reactions.PFK.lower_bound == -123.
            assert len(model._contexts) == 1
