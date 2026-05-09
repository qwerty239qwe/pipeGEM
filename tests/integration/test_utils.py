"""Tests for integration utility functions and supporting classes.

Covers:
- parse_predefined_threshold (all input variants)
- LinearDiscreteStrategy.transform
- integrator_factory registration and creation
- MBA called with explicit confidence sets (no data object)
- mCADRE score calculation helpers
"""
import numpy as np
import pandas as pd
import pytest
import cobra

from pipeGEM.integration.utils._thresholds import parse_predefined_threshold
from pipeGEM.integration.algo.CORDA import LinearDiscreteStrategy
from pipeGEM.integration._class import (
    integrator_factory,
    GIMME, EFlux, SPOT, RIPTiDePruning, RIPTiDeSampling, RIPTiDe,
    rFASTCORMICS, CORDA, mCADRE, MBA, INIT, iMAT,
)
from pipeGEM.integration.algo.mCADRE import calc_expr_score, calc_corr_score
from pipeGEM.integration.algo.MBA import apply_MBA
from pipeGEM.analysis.results.integration import MBA_Analysis


# ─────────────────────────────────────────────
# parse_predefined_threshold
# ─────────────────────────────────────────────

class TestParsePredefinedThreshold:

    def test_dict_input_returns_same_values(self):
        th = {"exp_th": 2.5, "non_exp_th": -1.5}
        result = parse_predefined_threshold(th, gene_data=None)
        assert result["exp_th"] == 2.5
        assert result["non_exp_th"] == -1.5

    def test_dict_input_th_result_is_the_dict(self):
        th = {"exp_th": 1.0, "non_exp_th": 0.0}
        result = parse_predefined_threshold(th)
        assert result["th_result"] is th

    def test_none_with_gene_data_uses_rfastcormics(self, ecoli_core_data):
        gene_data = pd.Series(ecoli_core_data["sample_0"])
        result = parse_predefined_threshold(
            None, gene_data=gene_data, threshold_type_if_none="rFASTCORMICS"
        )
        assert "exp_th" in result
        assert "non_exp_th" in result
        assert result["exp_th"] > result["non_exp_th"]

    def test_none_with_gene_data_uses_percentile(self, ecoli_core_data):
        gene_data = pd.Series(ecoli_core_data["sample_0"])
        result = parse_predefined_threshold(
            None, gene_data=gene_data,
            threshold_type_if_none="percentile", p=[75, 25],
        )
        assert result["exp_th"] >= result["non_exp_th"]

    def test_invalid_threshold_type_raises(self, ecoli_core_data):
        gene_data = pd.Series(ecoli_core_data["sample_0"])
        with pytest.raises(NotImplementedError):
            parse_predefined_threshold(
                None, gene_data=gene_data, threshold_type_if_none="unknown_method"
            )

    def test_analysis_object_input(self, ecoli_core_data):
        """Passing an analysis object preserves its exp_th / non_exp_th."""
        from pipeGEM.data import GeneData
        gene_data = GeneData(
            data=ecoli_core_data["sample_0"],
            data_transform=lambda x: np.log2(x),
            absent_expression=-np.inf,
        )
        th_obj = gene_data.get_threshold("percentile", p=[75, 25])
        result = parse_predefined_threshold(th_obj, gene_data=None)
        assert np.isclose(result["exp_th"], th_obj.exp_th)
        assert np.isclose(result["non_exp_th"], th_obj.non_exp_th)

    def test_nested_dict_with_analysis_objects(self, ecoli_core_data):
        """Nested dict: exp_th / non_exp_th values that are analysis objects."""
        from pipeGEM.data import GeneData
        gene_data = GeneData(
            data=ecoli_core_data["sample_0"],
            data_transform=lambda x: np.log2(x),
            absent_expression=-np.inf,
        )
        exp_obj = gene_data.get_threshold("percentile", p=75)
        nexp_obj = gene_data.get_threshold("percentile", p=25)
        th = {"exp_th": exp_obj, "non_exp_th": nexp_obj}
        result = parse_predefined_threshold(th)
        assert np.isclose(result["exp_th"], exp_obj.exp_th)
        assert np.isclose(result["non_exp_th"], nexp_obj.non_exp_th)


# ─────────────────────────────────────────────
# LinearDiscreteStrategy
# ─────────────────────────────────────────────

class TestLinearDiscreteStrategy:

    @pytest.fixture
    def strategy(self):
        scores = {"R_high": 3.0, "R_low": -3.0, "R_mid": 0.0, "R_inf": np.inf}
        return LinearDiscreteStrategy(exp_thres=2.0, nonexp_thres=-2.0,
                                      rxn_scores=scores)

    def test_score_at_exp_thres_maps_to_3(self):
        strat = LinearDiscreteStrategy(exp_thres=2.0, nonexp_thres=-2.0,
                                       rxn_scores={"R": 2.0})
        result = strat.transform()
        assert np.isclose(result["R"], 3.0)

    def test_score_at_nonexp_thres_maps_to_minus1(self):
        strat = LinearDiscreteStrategy(exp_thres=2.0, nonexp_thres=-2.0,
                                       rxn_scores={"R": -2.0})
        result = strat.transform()
        assert np.isclose(result["R"], -1.0)

    def test_midpoint_maps_to_1(self):
        """Mid-point between exp and nonexp maps to mid between 3 and -1 = 1."""
        strat = LinearDiscreteStrategy(exp_thres=2.0, nonexp_thres=-2.0,
                                       rxn_scores={"R": 0.0})
        result = strat.transform()
        assert np.isclose(result["R"], 1.0)

    def test_inf_score_maps_to_zero(self, strategy):
        result = strategy.transform()
        assert result["R_inf"] == 0

    def test_all_reactions_present_in_output(self, strategy):
        result = strategy.transform()
        assert set(result.keys()) == {"R_high", "R_low", "R_mid", "R_inf"}

    def test_monotone_increasing(self):
        scores = {"A": -3.0, "B": 0.0, "C": 3.0}
        strat = LinearDiscreteStrategy(exp_thres=2.0, nonexp_thres=-2.0,
                                       rxn_scores=scores)
        result = strat.transform()
        assert result["A"] < result["B"] < result["C"]

    def test_different_thresholds_give_different_mappings(self):
        scores = {"R": 1.0}
        s1 = LinearDiscreteStrategy(exp_thres=2.0, nonexp_thres=-2.0, rxn_scores=scores)
        s2 = LinearDiscreteStrategy(exp_thres=1.5, nonexp_thres=-1.5, rxn_scores=scores)
        r1, r2 = s1.transform(), s2.transform()
        assert r1["R"] != r2["R"]


# ─────────────────────────────────────────────
# integrator_factory
# ─────────────────────────────────────────────

class TestIntegratorFactory:
    """Check the factory has the expected integrators registered."""

    _expected = [
        ("GIMME", GIMME),
        ("EFlux", EFlux),
        ("RIPTiDePruning", RIPTiDePruning),
        ("RIPTiDeSampling", RIPTiDeSampling),
        ("RIPTiDe", RIPTiDe),
        ("rFASTCORMICS", rFASTCORMICS),
        ("CORDA", CORDA),
        ("mCADRE", mCADRE),
        ("MBA", MBA),
        ("INIT", INIT),
        ("iMAT", iMAT),
    ]

    @pytest.mark.parametrize("name,cls", _expected)
    def test_registered_integrator_creates_correct_type(self, name, cls):
        integrator = integrator_factory.create(name)
        assert isinstance(integrator, cls)

    def test_unknown_integrator_raises(self):
        with pytest.raises(KeyError):
            integrator_factory.create("NonExistentMethod")


# ─────────────────────────────────────────────
# MBA with explicit confidence sets (no data)
# ─────────────────────────────────────────────

class TestMBADirectConfSets:
    """MBA can be called with explicit high/medium confidence reaction lists
    instead of a data object, exercising the 'else' branch in apply_MBA."""

    def test_mba_with_direct_conf_sets(self, ecoli_core):
        rxn_ids = [r.id for r in ecoli_core.reactions]
        high_conf = rxn_ids[:10]
        medium_conf = rxn_ids[10:20]
        result = apply_MBA(
            model=ecoli_core,
            data=None,
            high_conf_rxn_ids=high_conf,
            medium_conf_rxn_ids=medium_conf,
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
            tolerance=1e-6,
            random_state=42,
        )
        assert isinstance(result, MBA_Analysis)

    def test_mba_direct_result_model_is_cobra_model(self, ecoli_core):
        rxn_ids = [r.id for r in ecoli_core.reactions]
        high_conf = rxn_ids[:5]
        medium_conf = rxn_ids[5:15]
        result = apply_MBA(
            model=ecoli_core,
            data=None,
            high_conf_rxn_ids=high_conf,
            medium_conf_rxn_ids=medium_conf,
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
        )
        assert isinstance(result.result_model, cobra.Model)

    def test_mba_direct_high_conf_reactions_kept(self, ecoli_core):
        rxn_ids = [r.id for r in ecoli_core.reactions]
        high_conf = rxn_ids[:5]
        medium_conf = rxn_ids[5:15]
        result = apply_MBA(
            model=ecoli_core,
            data=None,
            high_conf_rxn_ids=high_conf,
            medium_conf_rxn_ids=medium_conf,
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
        )
        kept_ids = {r.id for r in result.result_model.reactions}
        for r_id in high_conf:
            assert r_id in kept_ids

    def test_mba_direct_invalid_rxn_id_raises(self, ecoli_core):
        with pytest.raises(AssertionError):
            apply_MBA(
                model=ecoli_core,
                data=None,
                high_conf_rxn_ids=["BIOMASS_Ecoli_core_w_GAM"],
                medium_conf_rxn_ids=["NOT_A_REAL_REACTION"],
                protected_rxns=[],
            )

    def test_mba_threshold_analysis_none_with_direct_sets(self, ecoli_core):
        """When data=None, no threshold analysis is performed."""
        rxn_ids = [r.id for r in ecoli_core.reactions]
        result = apply_MBA(
            model=ecoli_core,
            data=None,
            high_conf_rxn_ids=rxn_ids[:5],
            medium_conf_rxn_ids=rxn_ids[5:10],
            protected_rxns=[],
        )
        assert result.threshold_analysis is None


# ─────────────────────────────────────────────
# mCADRE score helpers
# ─────────────────────────────────────────────

class TestMCADREHelpers:

    @pytest.fixture
    def scores_and_model(self, trivial_linear_model):
        rxn_ids = [r.id for r in trivial_linear_model.reactions]
        scores = {r: float(i) / len(rxn_ids) for i, r in enumerate(rxn_ids)}
        return trivial_linear_model, scores

    def test_calc_expr_score_returns_dict(self, scores_and_model):
        model, scores = scores_and_model
        from pipeGEM.data import GeneData

        class _FakeData:
            rxn_scores = scores

        result = calc_expr_score(
            _FakeData(), expr_th=0.8, nonexpr_th=0.2,
            absent_value=0.0,
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == set(scores.keys())

    def test_calc_expr_score_values_in_range(self, scores_and_model):
        model, scores = scores_and_model

        class _FakeData:
            rxn_scores = scores

        result = calc_expr_score(
            _FakeData(), expr_th=0.8, nonexpr_th=0.2, absent_value=0.0,
        )
        for v in result.values():
            assert -1 - 1e-6 <= v <= 1 + 1e-6

    def test_calc_expr_score_protected_rxns_set_to_1(self, scores_and_model):
        model, scores = scores_and_model
        protected = list(scores.keys())[:2]

        class _FakeData:
            rxn_scores = scores

        result = calc_expr_score(
            _FakeData(), expr_th=0.8, nonexpr_th=0.2,
            absent_value=0.0, protected_rxns=protected,
        )
        for r in protected:
            assert result[r] == 1.0

    def test_calc_corr_score_returns_dict_same_keys(self, scores_and_model):
        model, scores = scores_and_model

        class _FakeData:
            rxn_scores = scores

        expr_scores = calc_expr_score(
            _FakeData(), expr_th=0.8, nonexpr_th=0.2, absent_value=0.0,
        )
        corr_scores = calc_corr_score(model, expr_scores)
        assert isinstance(corr_scores, dict)
        assert set(corr_scores.keys()) == {r.id for r in model.reactions}

    def test_calc_corr_score_finite_values(self, scores_and_model):
        model, scores = scores_and_model

        class _FakeData:
            rxn_scores = scores

        expr_scores = calc_expr_score(
            _FakeData(), expr_th=0.8, nonexpr_th=0.2, absent_value=0.0,
        )
        corr_scores = calc_corr_score(model, expr_scores)
        for v in corr_scores.values():
            assert np.isfinite(v)
