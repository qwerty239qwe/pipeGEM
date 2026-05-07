import cobra
import numpy as np
import pytest

from pipeGEM.utils._manipulate import make_irrev_rxn


def _reaction_model():
    model = cobra.Model("split_test")
    a = cobra.Metabolite("a_c", compartment="c")
    b = cobra.Metabolite("b_c", compartment="c")

    rxn = cobra.Reaction("R")
    rxn.name = "Reversible reaction"
    rxn.lower_bound = -10
    rxn.upper_bound = 20
    rxn.gene_reaction_rule = "g1 and g2"
    rxn.notes = {"note": "kept"}
    rxn.annotation = {"bigg.reaction": "R"}
    rxn.add_metabolites({a: -1, b: 1})

    model.add_reactions([rxn])
    model.objective = rxn
    return model, a, b


def test_make_irrev_rxn_splits_reversible_reaction_and_removes_original():
    model, a, b = _reaction_model()

    new_rxns = make_irrev_rxn(model, "R", remove_original=True)

    assert [rxn.id for rxn in new_rxns] == ["_F_R", "_R_R"]
    assert "R" not in {rxn.id for rxn in model.reactions}

    forward = model.reactions.get_by_id("_F_R")
    backward = model.reactions.get_by_id("_R_R")

    assert forward.bounds == (0, 20)
    assert backward.bounds == (0, 10)
    assert forward.metabolites[a] == -1
    assert forward.metabolites[b] == 1
    assert backward.metabolites[a] == 1
    assert backward.metabolites[b] == -1
    assert forward.gene_reaction_rule == "g1 and g2"
    assert backward.gene_reaction_rule == "g1 and g2"
    assert forward.notes == {"note": "kept"}
    assert backward.annotation == {"bigg.reaction": "R"}
    assert np.isclose(forward.objective_coefficient, 1.0)
    assert np.isclose(backward.objective_coefficient, -1.0)


def test_make_irrev_rxn_preserves_positional_prefix_arguments():
    model, _, _ = _reaction_model()

    new_rxns = make_irrev_rxn(model, "R", True, False, "F_", "R_")

    assert [rxn.id for rxn in new_rxns] == ["F_R", "R_R"]
    assert "R" in {reaction.id for reaction in model.reactions}


def test_make_irrev_rxn_ignore_irreversible_skips_only_zero_capacity_direction():
    model = cobra.Model("forward_only")
    a = cobra.Metabolite("a_c", compartment="c")
    rxn = cobra.Reaction("R")
    rxn.lower_bound = 0
    rxn.upper_bound = 1000
    rxn.add_metabolites({a: -1})
    model.add_reactions([rxn])

    new_rxns = make_irrev_rxn(model, "R", ignore_irrev=True, remove_original=True)

    assert [reaction.id for reaction in new_rxns] == ["_F_R"]
    assert {reaction.id for reaction in model.reactions} == {"_F_R"}


def test_make_irrev_rxn_handles_backward_only_reaction():
    model = cobra.Model("backward_only")
    a = cobra.Metabolite("a_c", compartment="c")
    b = cobra.Metabolite("b_c", compartment="c")
    rxn = cobra.Reaction("R")
    rxn.lower_bound = -7
    rxn.upper_bound = 0
    rxn.add_metabolites({a: -1, b: 1})
    model.add_reactions([rxn])

    new_rxns = make_irrev_rxn(model, "R", ignore_irrev=True, remove_original=True)

    assert [reaction.id for reaction in new_rxns] == ["_R_R"]
    backward = model.reactions.get_by_id("_R_R")
    assert backward.bounds == (0, 7)
    assert backward.metabolites[a] == 1
    assert backward.metabolites[b] == -1
    assert "R" not in {reaction.id for reaction in model.reactions}


def test_make_irrev_rxn_raises_when_generated_ids_exist():
    model, _, _ = _reaction_model()
    model.add_reactions([cobra.Reaction("_F_R")])

    with pytest.raises(ValueError, match="Generated irreversible reaction IDs"):
        make_irrev_rxn(model, "R")
