"""Comprehensive tests for continuous integration algorithms.

Covers: GIMME, E-Flux, SPOT, RIPTiDe (pruning + sampling setup).
All tests use the ``trivial_linear_model`` fixture (function-scoped, fast)
and construct synthetic reaction scores directly, avoiding the overhead of
loading full GEM data.
"""
import numpy as np
import pandas as pd
import pytest
import cobra

from pipeGEM.integration.continuous.GIMME import apply_GIMME
from pipeGEM.integration.continuous.Eflux import apply_EFlux
from pipeGEM.integration.continuous.SPOT import apply_SPOT
from pipeGEM.integration.continuous.RIPTiDe import apply_RIPTiDe_pruning, apply_RIPTiDe_sampling
from pipeGEM.analysis.results.integration import (
    GIMMEAnalysis, EFluxAnalysis, SPOTAnalysis,
    RIPTiDePruningAnalysis, RIPTiDeSamplingAnalysis,
)


# ─────────────────────────────────────────────
# Shared score fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def rxn_ids(trivial_linear_model):
    return [r.id for r in trivial_linear_model.reactions]


@pytest.fixture
def low_scores(rxn_ids):
    """All scores well below 1.0 — GIMME will penalise every reaction."""
    return {r: 0.3 for r in rxn_ids}


@pytest.fixture
def high_scores(rxn_ids):
    """All scores above the default GIMME high_exp=1.0 — nothing penalised."""
    return {r: 2.0 for r in rxn_ids}


@pytest.fixture
def spread_scores(rxn_ids):
    """Scores spread linearly over (0.1, 0.9) — safe for EFlux (all > 0)
    and RIPTiDe (obj values land in [0, 1])."""
    n = len(rxn_ids)
    return {r: (i + 1) / (n + 2) for i, r in enumerate(rxn_ids)}


# ─────────────────────────────────────────────
# GIMME
# ─────────────────────────────────────────────

class TestGIMME:

    def test_returns_gimme_analysis(self, trivial_linear_model, low_scores):
        result = apply_GIMME(trivial_linear_model, low_scores, high_exp=1.0)
        assert isinstance(result, GIMMEAnalysis)

    def test_flux_result_is_dataframe_with_fluxes_column(self, trivial_linear_model, low_scores):
        result = apply_GIMME(trivial_linear_model, low_scores, high_exp=1.0)
        assert isinstance(result.flux_result, pd.DataFrame)
        assert "fluxes" in result.flux_result.columns

    def test_flux_result_covers_all_reactions(self, trivial_linear_model, low_scores):
        result = apply_GIMME(trivial_linear_model, low_scores, high_exp=1.0)
        expected_ids = {r.id for r in trivial_linear_model.reactions}
        assert expected_ids == set(result.flux_result.index)

    def test_no_flux_result_when_return_fluxes_false(self, trivial_linear_model, low_scores):
        result = apply_GIMME(trivial_linear_model, low_scores, high_exp=1.0,
                             return_fluxes=False)
        assert result.flux_result is None

    def test_result_model_none_when_remove_zero_fluxes_false(self, trivial_linear_model, low_scores):
        result = apply_GIMME(trivial_linear_model, low_scores, high_exp=1.0,
                             remove_zero_fluxes=False)
        assert result.result_model is None

    def test_result_model_is_cobra_model_when_remove_zero_fluxes_true(self, trivial_linear_model, low_scores):
        result = apply_GIMME(trivial_linear_model, low_scores, high_exp=1.0,
                             remove_zero_fluxes=True)
        # May or may not remove reactions, but must be a cobra.Model
        assert isinstance(result.result_model, cobra.Model)

    def test_result_model_reactions_subset_of_original(self, trivial_linear_model, low_scores):
        original_ids = {r.id for r in trivial_linear_model.reactions}
        result = apply_GIMME(trivial_linear_model, low_scores, high_exp=1.0,
                             remove_zero_fluxes=True)
        kept_ids = {r.id for r in result.result_model.reactions}
        assert kept_ids <= original_ids

    def test_low_expression_reactions_penalised(self, trivial_linear_model, low_scores):
        """All scores (0.3) < high_exp (1.0) → all eligible reactions penalised."""
        result = apply_GIMME(trivial_linear_model, low_scores, high_exp=1.0)
        # Every non-objective, non-protected reaction with score < high_exp should appear
        for coef in result.rxn_coefficents.values():
            assert coef > 0

    def test_reactions_above_high_exp_not_penalised(self, trivial_linear_model, high_scores):
        result = apply_GIMME(trivial_linear_model, high_scores, high_exp=1.0)
        assert len(result.rxn_coefficents) == 0

    def test_protected_reactions_not_penalised(self, trivial_linear_model, low_scores):
        protected = ["BIOMASS"]
        result = apply_GIMME(trivial_linear_model, low_scores, high_exp=1.0,
                             protected_rxns=protected)
        for p in protected:
            assert p not in result.rxn_coefficents

    def test_rxn_scores_stored_correctly(self, trivial_linear_model, low_scores):
        result = apply_GIMME(trivial_linear_model, low_scores, high_exp=1.0)
        assert result.rxn_scores == low_scores

    def test_original_model_not_mutated(self, trivial_linear_model, low_scores):
        n_before = len(trivial_linear_model.reactions)
        apply_GIMME(trivial_linear_model, low_scores, high_exp=1.0)
        assert len(trivial_linear_model.reactions) == n_before

    def test_penalty_magnitude_proportional_to_gap(self, trivial_linear_model, rxn_ids):
        """Higher gap (high_exp - score) → higher penalty coefficient."""
        scores_a = {r: 0.5 for r in rxn_ids}
        scores_b = {r: 0.1 for r in rxn_ids}
        high_exp = 1.0
        r_a = apply_GIMME(trivial_linear_model, scores_a, high_exp=high_exp)
        r_b = apply_GIMME(trivial_linear_model, scores_b, high_exp=high_exp)
        if r_a.rxn_coefficents and r_b.rxn_coefficents:
            avg_a = np.mean(list(r_a.rxn_coefficents.values()))
            avg_b = np.mean(list(r_b.rxn_coefficents.values()))
            assert avg_b > avg_a

    def test_log_stores_high_exp_and_obj_frac(self, trivial_linear_model, low_scores):
        result = apply_GIMME(trivial_linear_model, low_scores, high_exp=0.8, obj_frac=0.6)
        assert result.log["high_exp"] == 0.8
        assert result.log["obj_frac"] == 0.6


# ─────────────────────────────────────────────
# E-Flux
# ─────────────────────────────────────────────

class TestEFlux:

    def test_returns_eflux_analysis(self, trivial_linear_model, spread_scores):
        result = apply_EFlux(trivial_linear_model, spread_scores)
        assert isinstance(result, EFluxAnalysis)

    def test_rxn_bounds_covers_all_reactions(self, trivial_linear_model, spread_scores):
        result = apply_EFlux(trivial_linear_model, spread_scores)
        model_ids = {r.id for r in trivial_linear_model.reactions}
        assert model_ids == set(result.rxn_bounds.keys())

    def test_rxn_bounds_are_tuples(self, trivial_linear_model, spread_scores):
        result = apply_EFlux(trivial_linear_model, spread_scores)
        for bounds in result.rxn_bounds.values():
            assert isinstance(bounds, tuple) and len(bounds) == 2

    def test_flux_result_is_dataframe(self, trivial_linear_model, spread_scores):
        result = apply_EFlux(trivial_linear_model, spread_scores)
        assert isinstance(result.flux_result, pd.DataFrame)
        assert "fluxes" in result.flux_result.columns

    def test_no_flux_result_when_return_fluxes_false(self, trivial_linear_model, spread_scores):
        result = apply_EFlux(trivial_linear_model, spread_scores, return_fluxes=False)
        assert result.flux_result is None

    def test_result_model_is_cobra_model(self, trivial_linear_model, spread_scores):
        result = apply_EFlux(trivial_linear_model, spread_scores)
        assert isinstance(result.result_model, cobra.Model)

    def test_non_exchange_ub_within_max_ub(self, trivial_linear_model, spread_scores):
        max_ub = 50.0
        result = apply_EFlux(trivial_linear_model, spread_scores, max_ub=max_ub)
        for rxn in result.result_model.reactions:
            if rxn not in result.result_model.exchanges:
                assert rxn.upper_bound <= max_ub + 1e-9

    def test_exchange_bounds_unaffected(self, trivial_linear_model, spread_scores):
        original_ex_bounds = {r.id: r.bounds for r in trivial_linear_model.exchanges}
        result = apply_EFlux(trivial_linear_model, spread_scores)
        for rxn in result.result_model.exchanges:
            if rxn.id in original_ex_bounds:
                assert rxn.bounds == original_ex_bounds[rxn.id]

    def test_remove_zero_fluxes_yields_cobra_model(self, trivial_linear_model, spread_scores):
        result = apply_EFlux(trivial_linear_model, spread_scores, remove_zero_fluxes=True)
        assert isinstance(result.result_model, cobra.Model)

    def test_remove_zero_fluxes_result_subset_of_original(self, trivial_linear_model, spread_scores):
        original_ids = {r.id for r in trivial_linear_model.reactions}
        result = apply_EFlux(trivial_linear_model, spread_scores, remove_zero_fluxes=True)
        kept_ids = {r.id for r in result.result_model.reactions}
        assert kept_ids <= original_ids

    def test_remove_zero_fluxes_allows_no_protected_reactions(self, trivial_linear_model, spread_scores):
        result = apply_EFlux(trivial_linear_model, spread_scores, remove_zero_fluxes=True)
        assert isinstance(result.result_model, cobra.Model)

    def test_rxn_scores_stored(self, trivial_linear_model, spread_scores):
        result = apply_EFlux(trivial_linear_model, spread_scores)
        assert result.rxn_scores == spread_scores

    def test_original_model_not_mutated(self, trivial_linear_model, spread_scores):
        n_before = len(trivial_linear_model.reactions)
        apply_EFlux(trivial_linear_model, spread_scores)
        assert len(trivial_linear_model.reactions) == n_before

    def test_invalid_max_ub_raises(self, trivial_linear_model, spread_scores):
        with pytest.raises(AssertionError):
            apply_EFlux(trivial_linear_model, spread_scores, max_ub=-1.0)

    def test_invalid_min_lb_raises(self, trivial_linear_model, spread_scores):
        with pytest.raises(AssertionError):
            apply_EFlux(trivial_linear_model, spread_scores, min_lb=-0.5)

    def test_max_ub_smaller_than_min_lb_raises(self, trivial_linear_model, spread_scores):
        with pytest.raises(AssertionError):
            apply_EFlux(trivial_linear_model, spread_scores, max_ub=0.5, min_lb=1.0)

    def test_log_stores_parameters(self, trivial_linear_model, spread_scores):
        result = apply_EFlux(trivial_linear_model, spread_scores, max_ub=200.0, min_lb=0.01)
        assert result.log["max_ub"] == 200.0
        assert result.log["min_lb"] == 0.01


# ─────────────────────────────────────────────
# SPOT
# ─────────────────────────────────────────────

class TestSPOT:

    def test_returns_spot_analysis(self, trivial_linear_model, spread_scores):
        result = apply_SPOT(trivial_linear_model, spread_scores)
        assert isinstance(result, SPOTAnalysis)

    def test_flux_result_is_dataframe_when_return_fluxes_true(self, trivial_linear_model, spread_scores):
        result = apply_SPOT(trivial_linear_model, spread_scores, return_fluxes=True)
        assert isinstance(result.flux_result, pd.DataFrame)
        assert "fluxes" in result.flux_result.columns

    def test_no_flux_result_when_return_fluxes_false(self, trivial_linear_model, spread_scores):
        result = apply_SPOT(trivial_linear_model, spread_scores, return_fluxes=False)
        assert result.flux_result is None

    def test_result_model_none_without_remove(self, trivial_linear_model, spread_scores):
        result = apply_SPOT(trivial_linear_model, spread_scores, remove_zero_fluxes=False)
        assert result.result_model is None

    def test_result_model_cobra_model_with_remove(self, trivial_linear_model, spread_scores):
        result = apply_SPOT(trivial_linear_model, spread_scores, remove_zero_fluxes=True)
        assert isinstance(result.result_model, cobra.Model)

    def test_result_model_subset_of_original(self, trivial_linear_model, spread_scores):
        original_ids = {r.id for r in trivial_linear_model.reactions}
        result = apply_SPOT(trivial_linear_model, spread_scores, remove_zero_fluxes=True)
        if result.result_model is not None:
            kept_ids = {r.id for r in result.result_model.reactions}
            assert kept_ids <= original_ids

    def test_original_model_not_mutated(self, trivial_linear_model, spread_scores):
        n_before = len(trivial_linear_model.reactions)
        apply_SPOT(trivial_linear_model, spread_scores)
        assert len(trivial_linear_model.reactions) == n_before

    def test_protected_reactions_survive_pruning(self, trivial_linear_model, spread_scores):
        protected = ["BIOMASS"]
        result = apply_SPOT(trivial_linear_model, spread_scores,
                            remove_zero_fluxes=True, flux_threshold=1e6,
                            protected_rxns=protected)
        if result.result_model is not None:
            kept_ids = {r.id for r in result.result_model.reactions}
            for p in protected:
                assert p in kept_ids

    def test_protected_reaction_can_be_string(self, trivial_linear_model, spread_scores):
        result = apply_SPOT(trivial_linear_model, spread_scores,
                            remove_zero_fluxes=True, flux_threshold=1e6,
                            protected_rxns="BIOMASS")
        kept_ids = {r.id for r in result.result_model.reactions}
        assert "BIOMASS" in kept_ids
        assert result.log["protected_rxns"] == ["BIOMASS"]


# ─────────────────────────────────────────────
# RIPTiDe – Pruning
# ─────────────────────────────────────────────

class TestRIPTiDePruning:

    def test_returns_riptide_pruning_analysis(self, trivial_linear_model, spread_scores):
        result = apply_RIPTiDe_pruning(trivial_linear_model, spread_scores)
        assert isinstance(result, RIPTiDePruningAnalysis)

    def test_result_model_is_cobra_model(self, trivial_linear_model, spread_scores):
        result = apply_RIPTiDe_pruning(trivial_linear_model, spread_scores)
        assert isinstance(result.result_model, cobra.Model)

    def test_removed_ids_subset_of_original(self, trivial_linear_model, spread_scores):
        original_ids = {r.id for r in trivial_linear_model.reactions}
        result = apply_RIPTiDe_pruning(trivial_linear_model, spread_scores)
        assert set(result.removed_rxn_ids) <= original_ids

    def test_kept_plus_removed_equals_original(self, trivial_linear_model, spread_scores):
        original_ids = {r.id for r in trivial_linear_model.reactions}
        result = apply_RIPTiDe_pruning(trivial_linear_model, spread_scores)
        kept_ids = {r.id for r in result.result_model.reactions}
        removed_ids = set(result.removed_rxn_ids)
        assert kept_ids | removed_ids == original_ids
        assert len(kept_ids & removed_ids) == 0

    def test_protected_reactions_not_removed(self, trivial_linear_model, spread_scores):
        protected = ["BIOMASS"]
        result = apply_RIPTiDe_pruning(trivial_linear_model, spread_scores,
                                       protected_rxns=protected)
        assert "BIOMASS" not in result.removed_rxn_ids

    def test_result_model_feasible_with_protected_biomass(self, trivial_linear_model, spread_scores):
        result = apply_RIPTiDe_pruning(trivial_linear_model, spread_scores,
                                       protected_rxns=["BIOMASS"])
        sol = result.result_model.optimize()
        assert sol.status == "optimal"

    def test_obj_dict_values_in_unit_interval(self, trivial_linear_model, spread_scores):
        result = apply_RIPTiDe_pruning(trivial_linear_model, spread_scores)
        for v in result.obj_dict.values():
            assert -1e-9 <= v <= 1 + 1e-9

    def test_original_model_not_mutated(self, trivial_linear_model, spread_scores):
        n_before = len(trivial_linear_model.reactions)
        apply_RIPTiDe_pruning(trivial_linear_model, spread_scores)
        assert len(trivial_linear_model.reactions) == n_before

    def test_high_threshold_removes_at_least_as_many(self, trivial_linear_model, spread_scores):
        """A looser threshold prunes equal or more reactions."""
        r_low = apply_RIPTiDe_pruning(trivial_linear_model, spread_scores, threshold=1e-9)
        r_high = apply_RIPTiDe_pruning(trivial_linear_model, spread_scores, threshold=0.5)
        assert len(r_high.removed_rxn_ids) >= len(r_low.removed_rxn_ids)

    def test_log_stores_parameters(self, trivial_linear_model, spread_scores):
        result = apply_RIPTiDe_pruning(trivial_linear_model, spread_scores,
                                       threshold=1e-4, obj_frac=0.9)
        assert result.log["threshold"] == 1e-4
        assert result.log["obj_frac"] == 0.9

    def test_nan_scores_excluded(self, trivial_linear_model, rxn_ids):
        """NaN-scored reactions are excluded from the objective — no error."""
        scores = {r: float("nan") if i == 0 else 0.5 for i, r in enumerate(rxn_ids)}
        # Should not raise
        result = apply_RIPTiDe_pruning(trivial_linear_model, scores)
        assert isinstance(result, RIPTiDePruningAnalysis)


# ─────────────────────────────────────────────
# RIPTiDe – Sampling setup (do_sampling=False)
# ─────────────────────────────────────────────

class TestRIPTiDeSampling:

    def test_returns_riptide_sampling_analysis(self, trivial_linear_model, spread_scores):
        result = apply_RIPTiDe_sampling(trivial_linear_model, spread_scores,
                                        do_sampling=False)
        assert isinstance(result, RIPTiDeSamplingAnalysis)

    def test_no_sampling_result_when_do_sampling_false(self, trivial_linear_model, spread_scores):
        result = apply_RIPTiDe_sampling(trivial_linear_model, spread_scores,
                                        do_sampling=False)
        assert result.sampling_result is None

    def test_flux_result_none_without_sampling(self, trivial_linear_model, spread_scores):
        result = apply_RIPTiDe_sampling(trivial_linear_model, spread_scores,
                                        do_sampling=False)
        assert result.flux_result is None

    def test_log_stores_obj_frac(self, trivial_linear_model, spread_scores):
        result = apply_RIPTiDe_sampling(trivial_linear_model, spread_scores,
                                        do_sampling=False, obj_frac=0.7)
        assert result.log["obj_frac"] == 0.7

    def test_log_stores_sampling_method(self, trivial_linear_model, spread_scores):
        result = apply_RIPTiDe_sampling(trivial_linear_model, spread_scores,
                                        do_sampling=False, sampling_method="achr")
        assert result.log["sampling_method"] == "achr"

    def test_max_gw_too_small_raises(self, trivial_linear_model, spread_scores):
        """max_gw below the actual max score should raise ValueError."""
        with pytest.raises(ValueError):
            apply_RIPTiDe_sampling(trivial_linear_model, spread_scores,
                                   max_gw=0.001, do_sampling=False)

    def test_original_model_not_mutated(self, trivial_linear_model, spread_scores):
        n_before = len(trivial_linear_model.reactions)
        apply_RIPTiDe_sampling(trivial_linear_model, spread_scores, do_sampling=False)
        assert len(trivial_linear_model.reactions) == n_before

    def test_discard_inf_score_removes_infinite_entries(self, trivial_linear_model, rxn_ids):
        """Infinite scores are discarded when discard_inf_score=True.
        Finite scores must have variance so max_gw != min_gw (avoids divide-by-zero).
        """
        # Even indices → inf (discarded), odd indices → varied finite scores [0.2, 0.7]
        scores = {r: float("inf") if i % 2 == 0 else 0.2 + i * 0.05
                  for i, r in enumerate(rxn_ids)}
        result = apply_RIPTiDe_sampling(trivial_linear_model, scores,
                                        do_sampling=False, discard_inf_score=True)
        assert isinstance(result, RIPTiDeSamplingAnalysis)
