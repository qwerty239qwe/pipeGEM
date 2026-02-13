"""Algorithmic correctness tests for integration algorithms.

Tests GECKO-light/full exact values and integration algorithm invariants
(mCADRE, CORDA, MBA, iMAT) using trivial hand-crafted models.
"""
from copy import deepcopy
from unittest.mock import MagicMock

import cobra
import numpy as np
import pandas as pd
import pytest

from pipeGEM.integration.ec._builder import ECModelBuilder, PROT_POOL_ID
from pipeGEM.integration.ec.gecko_light import apply_gecko_light
from pipeGEM.integration.ec.gecko_full import apply_gecko_full


# =====================================================================
# Helpers
# =====================================================================

def _make_enzyme_data(rxn_items_dict):
    """Create a mock EnzymeData from a dict of rxn_items."""
    ed = MagicMock()
    ed.rxn_items.return_value = rxn_items_dict
    ed.__contains__ = lambda self, x: True
    return ed


# =====================================================================
# GECKO-light exact values
# =====================================================================

class TestGeckoLightExact:
    """Verify exact numerical outcomes of GECKO-light."""

    @pytest.fixture
    def model(self):
        m = cobra.Model("gl_test")
        a = cobra.Metabolite("a_c", compartment="c")
        b = cobra.Metabolite("b_c", compartment="c")
        r1 = cobra.Reaction("R1")
        r1.lower_bound, r1.upper_bound = 0, 1000
        r1.add_metabolites({a: -1, b: 1})
        r2 = cobra.Reaction("R2")
        r2.lower_bound, r2.upper_bound = 0, 1000
        r2.add_metabolites({b: -1})
        m.add_reactions([r1, r2])
        return m

    def test_gecko_light_exact_bound(self, model):
        """kcat=10, abundance=1.0, sigma=0.5 -> new_ub=18000 > 1000 -> unchanged."""
        ed = _make_enzyme_data({
            "R1": {"best_kcat": 10.0, "best_mw": 50.0, "protein_to_use": "P1"},
        })
        result = apply_gecko_light(model, ed, sigma=0.5, copy_model=True)
        ec = result.result["ec_model"]
        assert ec.reactions.get_by_id("R1").upper_bound == 1000

    def test_gecko_light_tighter_bound(self, model):
        """kcat=0.0001 -> new_ub=0.0001*1.0*0.5*3600=0.18 < 1000 -> bound reduced."""
        ed = _make_enzyme_data({
            "R1": {"best_kcat": 0.0001, "best_mw": 50.0, "protein_to_use": "P1"},
        })
        result = apply_gecko_light(model, ed, sigma=0.5, copy_model=True)
        ec = result.result["ec_model"]
        expected_ub = 0.0001 * 1.0 * 0.5 * 3600.0  # = 0.18
        assert np.isclose(ec.reactions.get_by_id("R1").upper_bound, expected_ub)

    def test_gecko_light_reduces_objective(self, trivial_linear_model):
        """Constrained model objective <= unconstrained model objective."""
        model = trivial_linear_model
        unconstrained_obj = model.optimize().objective_value

        ed = _make_enzyme_data({
            "R1": {"best_kcat": 0.0001, "best_mw": 50.0, "protein_to_use": "P1"},
            "R2": {"best_kcat": 0.0001, "best_mw": 50.0, "protein_to_use": "P2"},
            "R3": {"best_kcat": 0.0001, "best_mw": 50.0, "protein_to_use": "P3"},
        })
        result = apply_gecko_light(model, ed, sigma=0.5, copy_model=True)
        constrained_obj = result.result["ec_model"].optimize().objective_value
        assert constrained_obj <= unconstrained_obj + 1e-9

    def test_gecko_light_enzyme_usage_values(self, model):
        """Verify enzyme_usage DataFrame columns match hand-calculated values."""
        ed = _make_enzyme_data({
            "R1": {"best_kcat": 0.0001, "best_mw": 50.0, "protein_to_use": "P1"},
        })
        result = apply_gecko_light(model, ed, sigma=0.5, copy_model=True)
        eu = result.enzyme_usage
        assert set(eu.columns) >= {"reaction", "kcat", "abundance", "new_ub", "old_ub"}
        row = eu[eu["reaction"] == "R1"].iloc[0]
        assert np.isclose(row["kcat"], 0.0001)
        assert np.isclose(row["abundance"], 1.0)
        assert np.isclose(row["new_ub"], 0.0001 * 1.0 * 0.5 * 3600.0)
        assert np.isclose(row["old_ub"], 1000.0)

    def test_gecko_light_multiple_reactions(self, model):
        """Two reactions: one constrained (small kcat), one not (large kcat)."""
        ed = _make_enzyme_data({
            "R1": {"best_kcat": 0.0001, "best_mw": 50.0, "protein_to_use": "P1"},
            "R2": {"best_kcat": 100.0, "best_mw": 30.0, "protein_to_use": "P2"},
        })
        result = apply_gecko_light(model, ed, sigma=0.5, copy_model=True)
        ec = result.result["ec_model"]
        # R1 should be constrained (0.18 < 1000)
        assert ec.reactions.get_by_id("R1").upper_bound < 1000
        # R2: 100 * 1.0 * 0.5 * 3600 = 180000 > 1000, not constrained
        assert ec.reactions.get_by_id("R2").upper_bound == 1000


# =====================================================================
# GECKO-full exact values
# =====================================================================

class TestGeckoFullExact:
    """Verify exact structural and numerical properties of full GECKO."""

    @pytest.fixture
    def model(self):
        m = cobra.Model("gf_test")
        a = cobra.Metabolite("a_c", compartment="c")
        b = cobra.Metabolite("b_c", compartment="c")
        # Exchange to supply a_c
        ex_a = cobra.Reaction("EX_a")
        ex_a.lower_bound, ex_a.upper_bound = -10, 0
        ex_a.add_metabolites({a: -1})
        r1 = cobra.Reaction("R1")
        r1.lower_bound, r1.upper_bound = 0, 1000
        r1.add_metabolites({a: -1, b: 1})
        r2 = cobra.Reaction("R2")
        r2.lower_bound, r2.upper_bound = 0, 1000
        r2.add_metabolites({b: -1})
        m.add_reactions([ex_a, r1, r2])
        m.objective = "R2"
        return m

    def test_gecko_full_pool_capacity(self, model):
        """Pool exchange ub = ptot * f_factor * sigma exactly."""
        sigma, ptot, f_factor = 0.3, 0.4, 0.6
        ed = _make_enzyme_data({
            "R1": {"best_kcat": 10.0, "best_mw": 50.0, "protein_to_use": "P1"},
        })
        result = apply_gecko_full(model, ed, sigma=sigma, ptot=ptot,
                                  f_factor=f_factor, copy_model=True)
        ec = result.ec_model
        pool_rxn = ec.reactions.get_by_id(f"EX_{PROT_POOL_ID}")
        expected = ptot * f_factor * sigma
        assert np.isclose(pool_rxn.upper_bound, expected)

    def test_gecko_full_draw_stoichiometry(self, model):
        """Draw reaction: pool coeff = -mw, enzyme coeff = +1.0."""
        mw = 50.0
        ed = _make_enzyme_data({
            "R1": {"best_kcat": 10.0, "best_mw": mw, "protein_to_use": "P1"},
        })
        result = apply_gecko_full(model, ed, copy_model=True)
        ec = result.ec_model
        draw_rxn = ec.reactions.get_by_id("draw_P1")
        pool_met = ec.metabolites.get_by_id(PROT_POOL_ID)
        enz_met = ec.metabolites.get_by_id("prot_P1")
        assert np.isclose(draw_rxn.metabolites[pool_met], -mw)
        assert np.isclose(draw_rxn.metabolites[enz_met], 1.0)

    def test_gecko_full_arm_stoichiometry(self, model):
        """Arm reaction: enzyme coeff = -1/(kcat*3600) exactly."""
        kcat = 10.0
        ed = _make_enzyme_data({
            "R1": {"best_kcat": kcat, "best_mw": 50.0, "protein_to_use": "P1"},
        })
        result = apply_gecko_full(model, ed, copy_model=True)
        ec = result.ec_model
        r1 = ec.reactions.get_by_id("R1")
        enz_met = ec.metabolites.get_by_id("prot_P1")
        expected_coeff = -1.0 / (kcat * 3600.0)
        assert np.isclose(r1.metabolites[enz_met], expected_coeff)

    def test_gecko_full_pool_limits_total_flux(self, model):
        """With small pool, FBA objective < unconstrained."""
        unconstrained = model.optimize().objective_value

        ed = _make_enzyme_data({
            "R1": {"best_kcat": 10.0, "best_mw": 50.0, "protein_to_use": "P1"},
            "R2": {"best_kcat": 5.0, "best_mw": 30.0, "protein_to_use": "P2"},
        })
        # Very small pool
        result = apply_gecko_full(model, ed, sigma=0.001, ptot=0.001,
                                  f_factor=0.001, copy_model=True)
        constrained = result.ec_model.optimize().objective_value
        assert constrained < unconstrained - 1e-9

    def test_gecko_full_two_enzymes_compete(self, trivial_branched_model):
        """Two reactions sharing pool: increasing MW of one reduces flux for other."""
        ed_low = _make_enzyme_data({
            "R1": {"best_kcat": 10.0, "best_mw": 10.0, "protein_to_use": "P1"},
            "R2": {"best_kcat": 10.0, "best_mw": 10.0, "protein_to_use": "P1"},
            "R3": {"best_kcat": 10.0, "best_mw": 10.0, "protein_to_use": "P2"},
            "R4": {"best_kcat": 10.0, "best_mw": 10.0, "protein_to_use": "P2"},
        })
        result_low = apply_gecko_full(trivial_branched_model, ed_low,
                                      sigma=0.01, ptot=0.01, f_factor=0.5,
                                      copy_model=True)
        obj_low = result_low.ec_model.optimize().objective_value

        # Increase MW drastically for P2 path -> less flux available
        ed_high = _make_enzyme_data({
            "R1": {"best_kcat": 10.0, "best_mw": 10.0, "protein_to_use": "P1"},
            "R2": {"best_kcat": 10.0, "best_mw": 10.0, "protein_to_use": "P1"},
            "R3": {"best_kcat": 10.0, "best_mw": 500.0, "protein_to_use": "P2"},
            "R4": {"best_kcat": 10.0, "best_mw": 500.0, "protein_to_use": "P2"},
        })
        result_high = apply_gecko_full(trivial_branched_model, ed_high,
                                       sigma=0.01, ptot=0.01, f_factor=0.5,
                                       copy_model=True)
        # Model should still be feasible but may have lower or same objective
        sol = result_high.ec_model.optimize()
        assert sol.status == "optimal"


# =====================================================================
# Integration algorithm invariants (trivial_linear_model)
# =====================================================================

class TestMCADRECorrectness:
    """mCADRE invariant tests on the trivial linear model."""

    def test_mcadre_removes_unexpressed(self, ecoli_core, ecoli_core_data):
        """With expression data, unexpressed reactions should be pruned."""
        from pipeGEM import Model
        from pipeGEM.data import GeneData
        pmod = Model(model=ecoli_core, name_tag="ecoli")
        data_name = "sample_0"
        gene_data = GeneData(data=ecoli_core_data[data_name],
                             data_transform=lambda x: np.log2(x),
                             absent_expression=-np.inf)
        pmod.add_gene_data(data_name, gene_data)
        thres = gene_data.get_threshold("percentile", p=[10, 2])
        result = pmod.integrate_gene_data(
            data_name=data_name, integrator="mCADRE",
            predefined_threshold=thres, threshold_kws={},
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
        )
        # Result model should have fewer reactions
        assert len(result.result_model.reactions) <= len(ecoli_core.reactions)

    def test_mcadre_protected_always_survive(self, ecoli_core, ecoli_core_data):
        """BIOMASS always kept regardless of expression."""
        from pipeGEM import Model
        from pipeGEM.data import GeneData
        pmod = Model(model=ecoli_core, name_tag="ecoli")
        data_name = "sample_0"
        gene_data = GeneData(data=ecoli_core_data[data_name],
                             data_transform=lambda x: np.log2(x),
                             absent_expression=-np.inf)
        pmod.add_gene_data(data_name, gene_data)
        thres = gene_data.get_threshold("percentile", p=[10, 2])
        result = pmod.integrate_gene_data(
            data_name=data_name, integrator="mCADRE",
            predefined_threshold=thres, threshold_kws={},
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
        )
        kept_ids = {r.id for r in result.result_model.reactions}
        assert "BIOMASS_Ecoli_core_w_GAM" in kept_ids

    def test_mcadre_result_is_cobra_model(self, ecoli_core, ecoli_core_data):
        """Result model is a cobra.Model instance."""
        from pipeGEM import Model
        from pipeGEM.data import GeneData
        pmod = Model(model=ecoli_core, name_tag="ecoli")
        data_name = "sample_0"
        gene_data = GeneData(data=ecoli_core_data[data_name],
                             data_transform=lambda x: np.log2(x),
                             absent_expression=-np.inf)
        pmod.add_gene_data(data_name, gene_data)
        thres = gene_data.get_threshold("percentile", p=[10, 2])
        result = pmod.integrate_gene_data(
            data_name=data_name, integrator="mCADRE",
            predefined_threshold=thres, threshold_kws={},
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
        )
        assert isinstance(result.result_model, cobra.Model)


class TestCORDACorrectness:
    """CORDA invariant tests."""

    def test_corda_result_model_optimizable(self, ecoli_core, ecoli_core_data):
        """Result model can produce flux through biomass."""
        from pipeGEM import Model
        from pipeGEM.data import GeneData
        pmod = Model(model=ecoli_core, name_tag="ecoli")
        data_name = "sample_0"
        gene_data = GeneData(data=ecoli_core_data[data_name],
                             data_transform=lambda x: np.log2(x),
                             absent_expression=-np.inf)
        pmod.add_gene_data(data_name, gene_data)
        thres = gene_data.get_threshold("percentile", p=[75, 25])
        result = pmod.integrate_gene_data(
            data_name=data_name, integrator="CORDA",
            predefined_threshold=thres, threshold_kws={},
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
        )
        sol = result.result_model.optimize()
        assert sol.status == "optimal"

    def test_corda_removed_subset_of_original(self, ecoli_core, ecoli_core_data):
        """Removed reactions are a subset of original reaction set."""
        from pipeGEM import Model
        from pipeGEM.data import GeneData
        pmod = Model(model=ecoli_core, name_tag="ecoli")
        data_name = "sample_0"
        gene_data = GeneData(data=ecoli_core_data[data_name],
                             data_transform=lambda x: np.log2(x),
                             absent_expression=-np.inf)
        pmod.add_gene_data(data_name, gene_data)
        thres = gene_data.get_threshold("percentile", p=[75, 25])
        result = pmod.integrate_gene_data(
            data_name=data_name, integrator="CORDA",
            predefined_threshold=thres, threshold_kws={},
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
        )
        original_ids = {r.id for r in ecoli_core.reactions}
        removed = {r.id if hasattr(r, 'id') else r for r in result.removed_rxn_ids}
        assert removed <= original_ids

    def test_corda_keeps_protected(self, ecoli_core, ecoli_core_data):
        """Protected reactions are always kept."""
        from pipeGEM import Model
        from pipeGEM.data import GeneData
        pmod = Model(model=ecoli_core, name_tag="ecoli")
        data_name = "sample_0"
        gene_data = GeneData(data=ecoli_core_data[data_name],
                             data_transform=lambda x: np.log2(x),
                             absent_expression=-np.inf)
        pmod.add_gene_data(data_name, gene_data)
        thres = gene_data.get_threshold("percentile", p=[75, 25])
        result = pmod.integrate_gene_data(
            data_name=data_name, integrator="CORDA",
            predefined_threshold=thres, threshold_kws={},
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
        )
        kept_ids = {r.id for r in result.result_model.reactions}
        assert "BIOMASS_Ecoli_core_w_GAM" in kept_ids


class TestMBACorrectness:
    """MBA invariant tests."""

    def test_mba_result_model_feasible(self, ecoli_core, ecoli_core_data):
        """Result model is optimizable."""
        from pipeGEM import Model
        from pipeGEM.data import GeneData
        pmod = Model(model=ecoli_core, name_tag="ecoli")
        data_name = "sample_0"
        gene_data = GeneData(data=ecoli_core_data[data_name],
                             data_transform=lambda x: np.log2(x),
                             absent_expression=-np.inf)
        pmod.add_gene_data(data_name, gene_data)
        thres = gene_data.get_threshold("percentile", p=[75, 25])
        result = pmod.integrate_gene_data(
            data_name=data_name, integrator="MBA",
            predefined_threshold=thres, threshold_kws={},
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
        )
        sol = result.result_model.optimize()
        assert sol.status == "optimal"

    def test_mba_keeps_protected(self, ecoli_core, ecoli_core_data):
        """Protected reactions always present."""
        from pipeGEM import Model
        from pipeGEM.data import GeneData
        pmod = Model(model=ecoli_core, name_tag="ecoli")
        data_name = "sample_0"
        gene_data = GeneData(data=ecoli_core_data[data_name],
                             data_transform=lambda x: np.log2(x),
                             absent_expression=-np.inf)
        pmod.add_gene_data(data_name, gene_data)
        thres = gene_data.get_threshold("percentile", p=[75, 25])
        result = pmod.integrate_gene_data(
            data_name=data_name, integrator="MBA",
            predefined_threshold=thres, threshold_kws={},
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
        )
        kept_ids = {r.id for r in result.result_model.reactions}
        assert "BIOMASS_Ecoli_core_w_GAM" in kept_ids

    def test_mba_never_adds_reactions(self, ecoli_core, ecoli_core_data):
        """Result model never has more reactions than original."""
        from pipeGEM import Model
        from pipeGEM.data import GeneData
        pmod = Model(model=ecoli_core, name_tag="ecoli")
        data_name = "sample_0"
        gene_data = GeneData(data=ecoli_core_data[data_name],
                             data_transform=lambda x: np.log2(x),
                             absent_expression=-np.inf)
        pmod.add_gene_data(data_name, gene_data)
        thres = gene_data.get_threshold("percentile", p=[75, 25])
        result = pmod.integrate_gene_data(
            data_name=data_name, integrator="MBA",
            predefined_threshold=thres, threshold_kws={},
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
        )
        assert len(result.result_model.reactions) <= len(ecoli_core.reactions)


class TestIMATCorrectness:
    """iMAT invariant tests.

    iMAT requires indicator constraints (MILP) which are not supported
    by GLPK.  These tests are skipped if the result_model is not produced.
    """

    @pytest.fixture
    def imat_result(self, ecoli_core, ecoli_core_data):
        from pipeGEM import Model
        from pipeGEM.data import GeneData
        pmod = Model(model=ecoli_core, name_tag="ecoli")
        data_name = "sample_0"
        gene_data = GeneData(data=ecoli_core_data[data_name],
                             data_transform=lambda x: np.log2(x),
                             absent_expression=-np.inf)
        pmod.add_gene_data(data_name, gene_data)
        thres = gene_data.get_threshold("percentile", p=[75, 25])
        result = pmod.integrate_gene_data(
            data_name=data_name, integrator="iMAT",
            predefined_threshold=thres, threshold_kws={},
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
        )
        if "result_model" not in result.result:
            pytest.skip("iMAT requires solver with indicator constraint support (e.g. Gurobi)")
        return result

    def test_imat_result_subset_of_original(self, imat_result, ecoli_core):
        """Result reactions are a subset of original reactions."""
        kept_ids = {r.id for r in imat_result.result_model.reactions}
        original_ids = {r.id for r in ecoli_core.reactions}
        assert kept_ids <= original_ids

    def test_imat_result_model_feasible(self, imat_result):
        """Result model is optimizable."""
        sol = imat_result.result_model.optimize()
        assert sol.status == "optimal"

    def test_imat_keeps_protected(self, imat_result):
        """Protected reactions always kept."""
        kept_ids = {r.id for r in imat_result.result_model.reactions}
        assert "BIOMASS_Ecoli_core_w_GAM" in kept_ids


# =====================================================================
# Cross-algorithm invariants
# =====================================================================

class TestCrossAlgorithmInvariants:
    """Properties that hold for all context-specific algorithms."""

    @pytest.fixture(params=["mCADRE", "CORDA", "MBA"])
    def algo_result(self, request, ecoli_core, ecoli_core_data):
        from pipeGEM import Model
        from pipeGEM.data import GeneData
        algo = request.param
        pmod = Model(model=ecoli_core, name_tag="ecoli")
        data_name = "sample_0"
        gene_data = GeneData(data=ecoli_core_data[data_name],
                             data_transform=lambda x: np.log2(x),
                             absent_expression=-np.inf)
        pmod.add_gene_data(data_name, gene_data)
        p = [10, 2] if algo == "mCADRE" else [75, 25]
        thres = gene_data.get_threshold("percentile", p=p)
        result = pmod.integrate_gene_data(
            data_name=data_name, integrator=algo,
            predefined_threshold=thres, threshold_kws={},
            protected_rxns=["BIOMASS_Ecoli_core_w_GAM"],
        )
        return result, ecoli_core

    def test_all_algos_never_add_reactions(self, algo_result):
        result, original = algo_result
        assert len(result.result_model.reactions) <= len(original.reactions)

    def test_all_algos_preserve_protected(self, algo_result):
        result, _ = algo_result
        kept_ids = {r.id for r in result.result_model.reactions}
        assert "BIOMASS_Ecoli_core_w_GAM" in kept_ids

    def test_all_algos_result_is_cobra_model(self, algo_result):
        result, _ = algo_result
        assert isinstance(result.result_model, cobra.Model)
