"""Tests for enzyme-constrained model modules.

Covers:
- pipeGEM/integration/ec/_builder.py  (ECModelBuilder)
- pipeGEM/integration/ec/gecko_light.py  (apply_gecko_light)
- pipeGEM/integration/ec/gecko_full.py  (apply_gecko_full)
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
from pipeGEM.analysis.results.ec import GECKOLightAnalysis, GECKOFullAnalysis
from pipeGEM.data import EnzymeData


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def mini_model():
    """Minimal COBRA model with 2 reactions."""
    model = cobra.Model("mini")
    m_a = cobra.Metabolite("a_c", compartment="c")
    m_b = cobra.Metabolite("b_c", compartment="c")

    r1 = cobra.Reaction("R1")
    r1.lower_bound = 0
    r1.upper_bound = 1000
    r1.add_metabolites({m_a: -1, m_b: 1})

    r2 = cobra.Reaction("R2")
    r2.lower_bound = 0
    r2.upper_bound = 1000
    r2.add_metabolites({m_b: -1})

    model.add_reactions([r1, r2])
    return model


@pytest.fixture
def linear_flux_model():
    """Source -> R1 -> R2 objective model with a known unconstrained flux."""
    model = cobra.Model("linear_flux")
    a = cobra.Metabolite("a_c", compartment="c")
    b = cobra.Metabolite("b_c", compartment="c")

    ex_a = cobra.Reaction("EX_a")
    ex_a.lower_bound = -1000
    ex_a.upper_bound = 0
    ex_a.add_metabolites({a: -1})

    r1 = cobra.Reaction("R1")
    r1.lower_bound = 0
    r1.upper_bound = 1000
    r1.add_metabolites({a: -1, b: 1})

    r2 = cobra.Reaction("R2")
    r2.lower_bound = 0
    r2.upper_bound = 1000
    r2.add_metabolites({b: -1})

    model.add_reactions([ex_a, r1, r2])
    model.objective = "R2"
    return model


@pytest.fixture
def mock_enzyme_data():
    """Mock EnzymeData with rxn_items() returning kcat + MW info."""
    ed = MagicMock()
    ed.rxn_items.return_value = {
        "R1": {"best_kcat": 10.0, "best_mw": 50.0, "protein_to_use": "P1"},
        "R2": {"best_kcat": 5.0, "best_mw": 30.0, "protein_to_use": "P2"},
    }
    # Make `in` operator work for _check_gene_and_enzymes
    ed.__contains__ = lambda self, x: True
    return ed


# =====================================================================
# ECModelBuilder
# =====================================================================

class TestECModelBuilder:
    def test_add_protein_pool(self, mini_model):
        builder = ECModelBuilder(sigma=0.5, ptot=0.4, f_factor=0.6)
        prot_pool = builder.add_protein_pool(mini_model)

        assert prot_pool.id == PROT_POOL_ID
        assert PROT_POOL_ID in [m.id for m in mini_model.metabolites]
        # Check the exchange reaction
        ex_rxn = mini_model.reactions.get_by_id(f"EX_{PROT_POOL_ID}")
        assert ex_rxn is not None
        expected_ub = 0.4 * 0.6 * 0.5  # ptot * f_factor * sigma
        assert np.isclose(ex_rxn.upper_bound, expected_ub)

    def test_create_draw_reaction(self, mini_model):
        builder = ECModelBuilder()
        prot_pool = builder.add_protein_pool(mini_model)
        enz_met = builder.create_draw_reaction(mini_model, prot_pool, "P1", mw=50.0)

        assert enz_met.id == "prot_P1"
        assert "draw_P1" in [r.id for r in mini_model.reactions]
        draw_rxn = mini_model.reactions.get_by_id("draw_P1")
        # prot_pool consumed with stoich -mw, enz_met produced with +1
        assert draw_rxn.metabolites[prot_pool] == -50.0
        assert draw_rxn.metabolites[enz_met] == 1.0

    def test_create_draw_reaction_reuse_existing(self, mini_model):
        builder = ECModelBuilder()
        prot_pool = builder.add_protein_pool(mini_model)
        enz1 = builder.create_draw_reaction(mini_model, prot_pool, "P1", mw=50.0)
        enz2 = builder.create_draw_reaction(mini_model, prot_pool, "P1", mw=50.0)
        assert enz1 is enz2

    def test_add_protein_pool_reuses_existing_exchange(self, mini_model):
        builder = ECModelBuilder(sigma=0.5, ptot=0.4, f_factor=0.6)
        first_pool = builder.add_protein_pool(mini_model)
        second_pool = builder.add_protein_pool(mini_model)

        pool_exchanges = [rxn for rxn in mini_model.reactions if rxn.id == f"EX_{PROT_POOL_ID}"]
        assert first_pool is second_pool
        assert len(pool_exchanges) == 1
        assert np.isclose(pool_exchanges[0].upper_bound, 0.4 * 0.6 * 0.5)

    def test_create_draw_reaction_uses_existing_enzyme_metabolite(self, mini_model):
        builder = ECModelBuilder()
        prot_pool = builder.add_protein_pool(mini_model)
        existing = cobra.Metabolite("prot_P1", compartment="c")
        mini_model.add_metabolites([existing])

        enz_met = builder.create_draw_reaction(mini_model, prot_pool, "P1", mw=50.0)
        draw_rxn = mini_model.reactions.get_by_id("draw_P1")

        assert enz_met is existing
        assert draw_rxn.metabolites[prot_pool] == -50.0
        assert draw_rxn.metabolites[existing] == 1.0

    def test_create_arm_reaction(self, mini_model):
        builder = ECModelBuilder()
        prot_pool = builder.add_protein_pool(mini_model)
        enz_met = builder.create_draw_reaction(mini_model, prot_pool, "P1", mw=50.0)

        rxn = mini_model.reactions.get_by_id("R1")
        builder.create_arm_reaction(mini_model, rxn, enz_met, kcat=10.0)

        expected_coeff = -1.0 / (10.0 * 3600.0)
        assert np.isclose(rxn.metabolites[enz_met], expected_coeff)
        assert "R1" in builder.arm_reaction_ids

    def test_create_arm_reaction_kcat_zero_skipped(self, mini_model):
        builder = ECModelBuilder()
        prot_pool = builder.add_protein_pool(mini_model)
        enz_met = builder.create_draw_reaction(mini_model, prot_pool, "P1", mw=50.0)
        rxn = mini_model.reactions.get_by_id("R1")
        n_mets_before = len(rxn.metabolites)
        builder.create_arm_reaction(mini_model, rxn, enz_met, kcat=0.0)
        # Enzyme metabolite should NOT have been added
        assert len(rxn.metabolites) == n_mets_before

    def test_draw_and_arm_ids(self, mini_model):
        builder = ECModelBuilder()
        prot_pool = builder.add_protein_pool(mini_model)
        enz = builder.create_draw_reaction(mini_model, prot_pool, "P1", mw=50.0)
        rxn = mini_model.reactions.get_by_id("R1")
        builder.create_arm_reaction(mini_model, rxn, enz, kcat=10.0)

        assert "draw_P1" in builder.draw_reaction_ids
        assert "R1" in builder.arm_reaction_ids


# =====================================================================
# apply_gecko_light
# =====================================================================

class TestApplyGeckoLight:
    def test_basic_constraint(self, mini_model, mock_enzyme_data):
        result = apply_gecko_light(
            mini_model, mock_enzyme_data, sigma=0.5, copy_model=True,
        )
        assert isinstance(result, GECKOLightAnalysis)
        ec_model = result.result["ec_model"]
        # new_ub = kcat * 1.0 * sigma * 3600 = 10 * 1 * 0.5 * 3600 = 18000
        # Since 18000 > 1000 (original), bound should be unchanged for R1
        assert ec_model.reactions.get_by_id("R1").upper_bound == 1000

    def test_constraint_applied_when_new_ub_smaller(self, mini_model, mock_enzyme_data):
        """If kcat is small enough, the upper bound should be reduced."""
        mock_enzyme_data.rxn_items.return_value = {
            "R1": {"best_kcat": 0.001, "best_mw": 50.0, "protein_to_use": "P1"},
        }
        result = apply_gecko_light(
            mini_model, mock_enzyme_data, sigma=0.5, copy_model=True,
        )
        ec_model = result.result["ec_model"]
        # new_ub = 0.001 * (ptot * f_factor) * 0.5 * 3600 = 0.45
        assert ec_model.reactions.get_by_id("R1").upper_bound < 1000
        assert result.log["n_reactions_with_enzyme_data"] == 1
        assert result.log["n_bound_reductions"] == 1
        assert bool(result.enzyme_usage.loc[0, "bound_reduced"])

    def test_metadata_distinguishes_processed_rows_from_bound_reductions(
        self, mini_model, mock_enzyme_data
    ):
        mock_enzyme_data.rxn_items.return_value = {
            "R1": {"best_kcat": 10.0, "best_mw": 50.0, "protein_to_use": "P1"},
        }

        result = apply_gecko_light(
            mini_model, mock_enzyme_data, sigma=0.5, copy_model=True,
        )

        assert result.log["n_reactions_with_enzyme_data"] == 1
        assert result.log["n_bound_reductions"] == 0
        assert not bool(result.enzyme_usage.loc[0, "bound_reduced"])

    def test_global_protein_budget_affects_fallback_bound(self, mini_model, mock_enzyme_data):
        mock_enzyme_data.rxn_items.return_value = {
            "R1": {"best_kcat": 0.001, "best_mw": 50.0, "protein_to_use": "P1"},
        }
        result_low = apply_gecko_light(
            mini_model, mock_enzyme_data, sigma=0.5, ptot=0.1,
            f_factor=0.2, copy_model=True,
        )
        result_high = apply_gecko_light(
            mini_model, mock_enzyme_data, sigma=0.5, ptot=1.0,
            f_factor=1.0, copy_model=True,
        )

        low_ub = result_low.result["ec_model"].reactions.get_by_id("R1").upper_bound
        high_ub = result_high.result["ec_model"].reactions.get_by_id("R1").upper_bound
        assert low_ub < high_ub

    def test_copy_model_true_preserves_original(self, mini_model, mock_enzyme_data):
        original_ub = mini_model.reactions.get_by_id("R1").upper_bound
        apply_gecko_light(mini_model, mock_enzyme_data, sigma=0.5, copy_model=True)
        assert mini_model.reactions.get_by_id("R1").upper_bound == original_ub

    def test_copy_model_false_modifies_in_place(self, mini_model, mock_enzyme_data):
        mock_enzyme_data.rxn_items.return_value = {
            "R1": {"best_kcat": 0.001, "best_mw": 50.0, "protein_to_use": "P1"},
        }
        model_copy = deepcopy(mini_model)
        apply_gecko_light(model_copy, mock_enzyme_data, sigma=0.5, copy_model=False)
        assert model_copy.reactions.get_by_id("R1").upper_bound < 1000

    def test_protected_rxns_excluded(self, mini_model, mock_enzyme_data):
        mock_enzyme_data.rxn_items.return_value = {
            "R1": {"best_kcat": 0.001, "best_mw": 50.0, "protein_to_use": "P1"},
        }
        result = apply_gecko_light(
            mini_model, mock_enzyme_data, sigma=0.5,
            protected_rxns=["R1"], copy_model=True,
        )
        # R1 is protected, should not be constrained
        assert "R1" not in result.kcat_mapping

    def test_reaction_not_in_model_skipped(self, mini_model, mock_enzyme_data):
        mock_enzyme_data.rxn_items.return_value = {
            "NOT_IN_MODEL": {"best_kcat": 10.0, "best_mw": 50.0, "protein_to_use": "P1"},
        }
        result = apply_gecko_light(
            mini_model, mock_enzyme_data, sigma=0.5, copy_model=True,
        )
        assert "NOT_IN_MODEL" not in result.kcat_mapping

    def test_no_kcat_skipped(self, mini_model, mock_enzyme_data):
        mock_enzyme_data.rxn_items.return_value = {
            "R1": {"best_kcat": None, "best_mw": 50.0, "protein_to_use": "P1"},
        }
        result = apply_gecko_light(
            mini_model, mock_enzyme_data, sigma=0.5, copy_model=True,
        )
        assert "R1" not in result.kcat_mapping

    def test_flux_matches_applied_bound(self, linear_flux_model, mock_enzyme_data):
        """GECKO-light should reduce the optimum to the calculated kcat bound."""
        mock_enzyme_data.rxn_items.return_value = {
            "R1": {"best_kcat": 0.01, "best_mw": 10.0, "protein_to_use": "P1"},
        }
        result = apply_gecko_light(
            linear_flux_model,
            mock_enzyme_data,
            sigma=0.5,
            ptot=0.2,
            f_factor=0.5,
            copy_model=True,
        )
        ec_model = result.result["ec_model"]
        expected_bound = 0.01 * (0.2 * 0.5) * 0.5 * 3600.0

        solution = ec_model.optimize()

        assert solution.status == "optimal"
        assert np.isclose(ec_model.reactions.get_by_id("R1").upper_bound, expected_bound)
        assert np.isclose(solution.objective_value, expected_bound)


# =====================================================================
# apply_gecko_full
# =====================================================================

class TestApplyGeckoFull:
    def test_model_gains_protein_pool(self, mini_model, mock_enzyme_data):
        result = apply_gecko_full(
            mini_model, mock_enzyme_data, sigma=0.5, copy_model=True,
        )
        ec_model = result.ec_model
        met_ids = [m.id for m in ec_model.metabolites]
        assert PROT_POOL_ID in met_ids

    def test_draw_and_arm_reactions_created(self, mini_model, mock_enzyme_data):
        result = apply_gecko_full(
            mini_model, mock_enzyme_data, sigma=0.5, copy_model=True,
        )
        assert len(result.draw_reactions) > 0
        assert len(result.arm_reactions) > 0

    def test_invalid_kcat_skipped(self, mini_model, mock_enzyme_data):
        mock_enzyme_data.rxn_items.return_value = {
            "R1": {"best_kcat": np.nan, "best_mw": 50.0, "protein_to_use": "P1"},
        }
        result = apply_gecko_full(
            mini_model, mock_enzyme_data, copy_model=True,
        )
        assert len(result.arm_reactions) == 0

    def test_invalid_mw_skipped(self, mini_model, mock_enzyme_data):
        mock_enzyme_data.rxn_items.return_value = {
            "R1": {"best_kcat": 10.0, "best_mw": np.nan, "protein_to_use": "P1"},
        }
        result = apply_gecko_full(
            mini_model, mock_enzyme_data, copy_model=True,
        )
        assert len(result.arm_reactions) == 0

    def test_result_object_populated(self, mini_model, mock_enzyme_data):
        result = apply_gecko_full(
            mini_model, mock_enzyme_data, sigma=0.5, ptot=0.4, f_factor=0.6,
            copy_model=True,
        )
        assert isinstance(result, GECKOFullAnalysis)
        assert result.protein_pool_id == PROT_POOL_ID
        assert result.ec_model is not None
        assert result.log["sigma"] == 0.5

    def test_protein_pool_limits_flux_to_calculated_capacity(self, linear_flux_model, mock_enzyme_data):
        """Full GECKO should limit flux through MW, kcat, and the protein pool."""
        mock_enzyme_data.rxn_items.return_value = {
            "R1": {"best_kcat": 1.0, "best_mw": 10.0, "protein_to_use": "P1"},
        }
        sigma = 0.5
        ptot = 0.2
        f_factor = 0.5

        result = apply_gecko_full(
            linear_flux_model,
            mock_enzyme_data,
            sigma=sigma,
            ptot=ptot,
            f_factor=f_factor,
            copy_model=True,
        )
        ec_model = result.ec_model
        expected_pool_ub = ptot * f_factor * sigma
        expected_flux = expected_pool_ub * 1.0 * 3600.0 / 10.0

        solution = ec_model.optimize()
        pool_exchange = ec_model.reactions.get_by_id(f"EX_{PROT_POOL_ID}")
        r1 = ec_model.reactions.get_by_id("R1")
        enz_met = ec_model.metabolites.get_by_id("prot_P1")

        assert solution.status == "optimal"
        assert np.isclose(pool_exchange.upper_bound, expected_pool_ub)
        assert np.isclose(r1.metabolites[enz_met], -1.0 / 3600.0)
        assert np.isclose(solution.objective_value, expected_flux)
        assert np.isclose(solution.fluxes[f"EX_{PROT_POOL_ID}"], expected_pool_ub)

    def test_full_gecko_uses_dlkcat_fallback_from_enzyme_data(self, linear_flux_model):
        enzyme_df = pd.DataFrame({
            "MW": [10.0],
            "Kcat": [np.nan],
            "DLKcat": [1.0],
            "Sequence": ["ACDEF"],
            "Reaction": ["R1"],
        }, index=["g1"])
        enzyme_data = EnzymeData(enzyme_df, rxn_id_col="Reaction")
        with pytest.warns(UserWarning):
            enzyme_data.align(linear_flux_model, check_and_raise=False, run_DLKcat=False)

        result = apply_gecko_full(
            linear_flux_model,
            enzyme_data,
            sigma=0.5,
            ptot=0.2,
            f_factor=0.5,
            copy_model=True,
        )

        expected_flux = 0.5 * 0.2 * 0.5 * 3600.0 / 10.0
        solution = result.ec_model.optimize()

        assert solution.status == "optimal"
        assert np.isclose(solution.objective_value, expected_flux)

    def test_full_gecko_complex_branched_model_uses_lowest_protein_cost_path(self):
        """A shared-enzyme pathway should outcompete a higher protein-cost branch."""
        model = cobra.Model("complex_ec")
        a = cobra.Metabolite("a_c", compartment="c")
        b = cobra.Metabolite("b_c", compartment="c")
        c = cobra.Metabolite("c_c", compartment="c")
        d = cobra.Metabolite("d_c", compartment="c")

        ex_a = cobra.Reaction("EX_a")
        ex_a.lower_bound = -1000
        ex_a.upper_bound = 0
        ex_a.add_metabolites({a: -1})

        fast_1 = cobra.Reaction("FAST_1")
        fast_1.lower_bound = 0
        fast_1.upper_bound = 1000
        fast_1.add_metabolites({a: -1, b: 1})

        fast_2 = cobra.Reaction("FAST_2")
        fast_2.lower_bound = 0
        fast_2.upper_bound = 1000
        fast_2.add_metabolites({b: -1, d: 1})

        slow_1 = cobra.Reaction("SLOW_1")
        slow_1.lower_bound = 0
        slow_1.upper_bound = 1000
        slow_1.add_metabolites({a: -1, c: 1})

        slow_2 = cobra.Reaction("SLOW_2")
        slow_2.lower_bound = 0
        slow_2.upper_bound = 1000
        slow_2.add_metabolites({c: -1, d: 1})

        link = cobra.Reaction("B_C_LINK")
        link.lower_bound = -1000
        link.upper_bound = 1000
        link.add_metabolites({b: -1, c: 1})

        dm_d = cobra.Reaction("DM_d")
        dm_d.lower_bound = 0
        dm_d.upper_bound = 1000
        dm_d.add_metabolites({d: -1})

        model.add_reactions([ex_a, fast_1, fast_2, slow_1, slow_2, link, dm_d])
        model.objective = "DM_d"

        enzyme_data = MagicMock()
        enzyme_data.rxn_items.return_value = {
            "FAST_1": {"best_kcat": 10.0, "best_mw": 10.0, "protein_to_use": "P_FAST"},
            "FAST_2": {"best_kcat": 10.0, "best_mw": 10.0, "protein_to_use": "P_FAST"},
            "SLOW_1": {"best_kcat": 1.0, "best_mw": 100.0, "protein_to_use": "P_SLOW"},
            "SLOW_2": {"best_kcat": 1.0, "best_mw": 100.0, "protein_to_use": "P_SLOW"},
            "B_C_LINK": {"best_kcat": 5.0, "best_mw": 10.0, "protein_to_use": "P_LINK"},
        }

        result = apply_gecko_full(
            model,
            enzyme_data,
            sigma=1.0,
            ptot=0.1,
            f_factor=0.5,
            copy_model=True,
        )

        ec_model = result.ec_model
        solution = ec_model.optimize()
        pool_ub = 1.0 * 0.1 * 0.5
        expected_fast_flux = pool_ub * (10.0 * 3600.0) / (2 * 10.0)

        assert solution.status == "optimal"
        assert np.isclose(solution.objective_value, expected_fast_flux)
        assert np.isclose(solution.fluxes["FAST_1"], expected_fast_flux)
        assert np.isclose(solution.fluxes["FAST_2"], expected_fast_flux)
        assert np.isclose(solution.fluxes["SLOW_1"], 0.0)
        assert np.isclose(solution.fluxes["SLOW_2"], 0.0)
        assert "B_C_LINK" not in {rxn.id for rxn in ec_model.reactions}
        assert {"_F_B_C_LINK", "_R_B_C_LINK"} <= {rxn.id for rxn in ec_model.reactions}
        assert set(result.draw_reactions) == {"draw_P_FAST", "draw_P_SLOW", "draw_P_LINK"}
        assert set(result.arm_reactions) == {
            "FAST_1", "FAST_2", "SLOW_1", "SLOW_2", "_F_B_C_LINK", "_R_B_C_LINK",
        }
        assert result.log["n_enzyme_constraints"] == 6
        assert np.isclose(solution.fluxes[f"EX_{PROT_POOL_ID}"], pool_ub)

    def test_reversible_reaction_is_split_before_enzyme_constraint(self):
        """Backward flux should consume enzyme instead of producing it."""
        model = cobra.Model("reverse_flux")
        a = cobra.Metabolite("a_c", compartment="c")
        b = cobra.Metabolite("b_c", compartment="c")

        ex_b = cobra.Reaction("EX_b")
        ex_b.lower_bound = -1000
        ex_b.upper_bound = 0
        ex_b.add_metabolites({b: -1})

        rrev = cobra.Reaction("RREV")
        rrev.lower_bound = -1000
        rrev.upper_bound = 1000
        rrev.add_metabolites({a: -1, b: 1})

        dm_a = cobra.Reaction("DM_a")
        dm_a.lower_bound = 0
        dm_a.upper_bound = 1000
        dm_a.add_metabolites({a: -1})

        model.add_reactions([ex_b, rrev, dm_a])
        model.objective = "DM_a"

        enzyme_data = MagicMock()
        enzyme_data.rxn_items.return_value = {
            "RREV": {"best_kcat": 1.0, "best_mw": 10.0, "protein_to_use": "P1"},
        }
        sigma = 0.5
        ptot = 0.2
        f_factor = 0.5

        result = apply_gecko_full(
            model,
            enzyme_data,
            sigma=sigma,
            ptot=ptot,
            f_factor=f_factor,
            copy_model=True,
        )

        ec_model = result.ec_model
        solution = ec_model.optimize()
        enz_met = ec_model.metabolites.get_by_id("prot_P1")
        expected_pool_ub = sigma * ptot * f_factor
        expected_flux = expected_pool_ub * 3600.0 / 10.0

        assert solution.status == "optimal"
        assert "RREV" not in {rxn.id for rxn in ec_model.reactions}
        assert {"_F_RREV", "_R_RREV"} <= {rxn.id for rxn in ec_model.reactions}
        assert result.log["n_enzyme_constraints"] == 2
        assert set(result.arm_reactions) == {"_F_RREV", "_R_RREV"}
        assert np.isclose(ec_model.reactions.get_by_id("_F_RREV").metabolites[enz_met], -1.0 / 3600.0)
        assert np.isclose(ec_model.reactions.get_by_id("_R_RREV").metabolites[enz_met], -1.0 / 3600.0)
        assert np.isclose(solution.fluxes["_F_RREV"], 0.0)
        assert np.isclose(solution.fluxes["_R_RREV"], expected_flux)
        assert np.isclose(solution.objective_value, expected_flux)
        assert np.isclose(solution.fluxes[f"EX_{PROT_POOL_ID}"], expected_pool_ub)
