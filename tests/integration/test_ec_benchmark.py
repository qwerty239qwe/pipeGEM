"""Hand-solvable GECKO benchmark.

Small enough to solve on paper, but it exercises non-unit stoichiometry, a
promiscuous enzyme with two kcats, a reversible reaction that runs backward
mid-pathway, a capped branch (mixed optimum) and every limiting regime.

    EX_A: -> A                 (ub = 10)
    R1  : A -> 2 B             E1  kcat 1    MW 36 kDa
    R2  : B -> P   (ub = 4)    E2  kcat 2    MW 72 kDa
    R3  : A -> P               E2  kcat 0.5  (same enzyme, other kcat)
    RREV: Q <=> P              E4  kcat 1    MW 36 kDa  (runs P -> Q)
    DM_Q: Q ->                 objective

Protein cost per unit flux c = MW[g/mmol] / (kcat * 3600):
c1 = c2 = c4 = 0.01, c3 = 0.04.  Per unit of Q:
    branch 1 (R1/R2): 0.5 A, pool 0.5*c1 + c2 + c4 = 0.025, capped at Q = 4
    branch 2 (R3)   : 1.0 A, pool c3 + c4           = 0.05
Branch 1 is cheaper in *both* resources, so the LP optimum is greedy:
fill branch 1 to its cap, spill the rest into branch 2.
"""
import cobra
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

import pipeGEM as pg
from pipeGEM.data import EnzymeData
from pipeGEM.data.data import ProteinAbundanceData
from pipeGEM.integration.ec import apply_gecko_full, apply_gecko_light
from pipeGEM.integration.ec._builder import PROT_POOL_ID

UPTAKE, CAP = 10.0, 4.0
COST_1, COST_2 = 0.025, 0.05
F_FACTOR, SIGMA = 0.5, 0.5  # pool = ptot / 4

ENZYME_ROWS = pd.DataFrame({
    "Reaction": ["R1", "R2", "R3", "RREV"],
    "Protein": ["E1", "E2", "E2", "E4"],
    "Kcat": [1.0, 2.0, 0.5, 1.0],
    "MW": [36_000.0, 72_000.0, 72_000.0, 36_000.0],  # Da, as EnzymeData stores it
}, index=["g1", "g2", "g2", "g4"])


def expected_q(pool):
    """Closed-form optimum of the benchmark for a given pool capacity."""
    q1 = min(CAP, pool / COST_1, UPTAKE / 0.5)
    q2 = min((pool - COST_1 * q1) / COST_2, UPTAKE - 0.5 * q1)
    return q1 + q2


@pytest.fixture
def bench():
    model = cobra.Model("ec_benchmark")
    A, B, P, Q = (cobra.Metabolite(m, compartment="c") for m in "ABPQ")

    def rxn(rid, stoich, lb, ub, gpr=""):
        r = cobra.Reaction(rid, lower_bound=lb, upper_bound=ub)
        r.add_metabolites(stoich)
        r.gene_reaction_rule = gpr
        return r

    model.add_reactions([
        rxn("EX_A", {A: 1}, 0, UPTAKE),
        rxn("R1", {A: -1, B: 2}, 0, 1000, "g1"),
        rxn("R2", {B: -1, P: 1}, 0, CAP, "g2"),
        rxn("R3", {A: -1, P: 1}, 0, 1000, "g2"),
        rxn("RREV", {Q: -1, P: 1}, -1000, 1000, "g4"),
        rxn("DM_Q", {Q: -1}, 0, 1000),
    ])
    model.objective = "DM_Q"
    return model


def _enzyme_data(model):
    ed = EnzymeData(ENZYME_ROWS, prot_id_col="Protein", rxn_id_col="Reaction")
    ed.align(model, check_and_raise=False, run_DLKcat=False)
    return ed


def _mock_enzyme_data(_model):
    ed = MagicMock()
    ed.rxn_items.return_value = {
        row.Reaction: {"best_kcat": row.Kcat, "best_mw": row.MW, "protein_to_use": row.Protein}
        for row in ENZYME_ROWS.itertuples()
    }
    return ed


@pytest.fixture(params=[_enzyme_data, _mock_enzyme_data], ids=["EnzymeData", "mock"])
def enzyme_data(request, bench):
    return request.param(bench)


def _full(bench, enzyme_data, ptot, **kwargs):
    return apply_gecko_full(bench, enzyme_data, ptot=ptot, f_factor=F_FACTOR, sigma=SIGMA, **kwargs)


# =====================================================================
# Full GECKO
# =====================================================================

@pytest.mark.parametrize("ptot, regime", [
    (0.2, "pool-limited, branch 1 only"),
    (0.4, "branch 1 exactly at cap"),
    (1.2, "mixed: cap + pool"),
    (4.0, "substrate-limited"),
])
def test_full_matches_closed_form_in_every_regime(bench, enzyme_data, ptot, regime):
    result = _full(bench, enzyme_data, ptot)
    assert np.isclose(result.ec_model.slim_optimize(), expected_q(ptot / 4)), regime


def test_full_mixed_optimum_fluxes_and_enzyme_usage(bench, enzyme_data):
    """At pool = 0.3: Q1 = Q2 = 4 -> every flux and draw is known exactly."""
    ec = _full(bench, enzyme_data, ptot=1.2).ec_model
    fluxes = ec.optimize().fluxes

    expected = {
        "EX_A": 6.0, "R1": 2.0, "R2": 4.0, "R3": 4.0,
        "_F_RREV": 0.0, "_R_RREV": 8.0, "DM_Q": 8.0,
        f"EX_{PROT_POOL_ID}": 0.3,           # pool fully used
        "draw_E1": 2.0 / 3600,               # v / (kcat * 3600)
        "draw_E2": 4.0 / 7200 + 4.0 / 1800,  # shared by R2 and R3
        "draw_E4": 8.0 / 3600,
    }
    for rid, value in expected.items():
        assert np.isclose(fluxes[rid], value), rid


def test_full_promiscuous_enzyme_shares_one_draw_with_per_rxn_kcat(bench, enzyme_data):
    result = _full(bench, enzyme_data, ptot=1.2)
    ec = result.ec_model
    e2 = ec.metabolites.get_by_id("prot_E2")

    assert sorted(result.draw_reactions) == ["draw_E1", "draw_E2", "draw_E4"]
    assert np.isclose(ec.reactions.R2.metabolites[e2], -1 / (2.0 * 3600))
    assert np.isclose(ec.reactions.R3.metabolites[e2], -1 / (0.5 * 3600))
    pool = ec.metabolites.get_by_id(PROT_POOL_ID)
    assert np.isclose(ec.reactions.draw_E2.metabolites[pool], -72.0)  # g/mmol


def test_full_protected_reversible_rxn_stays_unconstrained(bench, enzyme_data):
    """Without E4 cost: branch 1 costs 0.015, branch 2 0.04 -> 4 + 0.24/0.04 = 10."""
    result = _full(bench, enzyme_data, ptot=1.2, protected_rxns=["RREV"])
    ec = result.ec_model
    assert "RREV" in ec.reactions and "_R_RREV" not in ec.reactions
    assert np.isclose(ec.slim_optimize(), 10.0)


def test_full_objective_on_split_reaction_is_preserved(bench, enzyme_data):
    """Maximising backward RREV must give the same optimum as maximising DM_Q."""
    bench.objective = {bench.reactions.RREV: -1}
    ec = _full(bench, enzyme_data, ptot=1.2).ec_model
    assert np.isclose(ec.slim_optimize(), 8.0)


def test_full_copy_model_flag(bench, enzyme_data):
    original_ids = [r.id for r in bench.reactions]
    _full(bench, enzyme_data, ptot=1.2, copy_model=True)
    assert [r.id for r in bench.reactions] == original_ids

    _full(bench, enzyme_data, ptot=1.2, copy_model=False)
    assert "_R_RREV" in bench.reactions and PROT_POOL_ID in bench.metabolites


# =====================================================================
# GECKO light
# =====================================================================

def test_light_with_custom_abundance_column(bench, enzyme_data):
    """cap = kcat * abundance * sigma * 3600; abundance_col must be honoured."""
    abund = ProteinAbundanceData(
        pd.DataFrame({"conc": [1e-3, 1e-3, 1e-3]}, index=["E1", "E2", "E4"]),
        abundance_col="conc",
    )
    result = apply_gecko_light(bench, enzyme_data, protein_abundance=abund, sigma=SIGMA)
    ec = result.ec_model

    for rid, bounds in {"R1": (0, 1.8), "R2": (0, 3.6), "R3": (0, 0.9),
                        "RREV": (-1.8, 1.8)}.items():
        assert np.allclose(ec.reactions.get_by_id(rid).bounds, bounds), rid
    # R1 (1.8 -> 3.6 B -> R2 3.6) + R3 0.9 = 4.5 P, but RREV caps Q at 1.8
    assert np.isclose(ec.slim_optimize(), 1.8)


# =====================================================================
# pipeGEM.Model entry point
# =====================================================================

def test_pg_model_integrate_enzyme_data_end_to_end(bench):
    model = pg.Model(name_tag="bench", model=bench)
    model.add_enzyme_data(EnzymeData(ENZYME_ROWS, prot_id_col="Protein", rxn_id_col="Reaction"),
                          check_and_raise=False, run_DLKcat=False)
    result = model.integrate_enzyme_data(method="GECKOFull", ptot=1.2,
                                         f_factor=F_FACTOR, sigma=SIGMA)
    assert np.isclose(result.ec_model.slim_optimize(), 8.0)

    model.add_protein_abundance_data("prot", ProteinAbundanceData(
        pd.DataFrame({"abundance": [1e-3] * 3}, index=["E1", "E2", "E4"])))
    light = model.integrate_enzyme_data("prot", method="GECKOLight", sigma=SIGMA)
    assert np.isclose(light.ec_model.slim_optimize(), 1.8)


def test_pg_model_unknown_abundance_name_raises(bench):
    model = pg.Model(name_tag="bench", model=bench)
    model.add_enzyme_data(EnzymeData(ENZYME_ROWS, prot_id_col="Protein", rxn_id_col="Reaction"),
                          check_and_raise=False, run_DLKcat=False)
    with pytest.raises(KeyError, match="typo"):
        model.integrate_enzyme_data("typo", method="GECKOLight")
