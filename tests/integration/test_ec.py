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
        # new_ub = 0.001 * 1.0 * 0.5 * 3600 = 1.8
        assert ec_model.reactions.get_by_id("R1").upper_bound < 1000

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
