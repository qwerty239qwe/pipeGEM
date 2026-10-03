"""Check scores against arithmetic and Boolean oracles, never mapper outputs.

For complete finite measurements and min/max scoring, score >= t iff the
Boolean rule is true with genes having expression >= t. Truth tables and COBRA's knockout evaluator
therefore provide a separate algorithm for expected scores. Other reductions
use explicit arithmetic with Python's min/max/sum/mean, rather than NumPy or
the mapper's AST traversal. Expected answers are computed before mapping.
"""

from ast import Constant
from itertools import product
from math import sqrt
from pathlib import Path
from statistics import mean
import subprocess
import sys
from types import SimpleNamespace

import cobra
import numpy as np
import pandas as pd
import pytest

from pipeGEM import Model
from pipeGEM.analysis import RxnMapper
from pipeGEM.analysis._reducing import MergedReaction
from pipeGEM.data import GeneData


# These Boolean expressions are written independently of COBRA's GPR parser.
BOOLEAN_RULES = [
    ("g1 and (g2 or g3)", lambda a, b, c: a and (b or c)),
    ("g1 and g2 or g3", lambda a, b, c: (a and b) or c),
    ("(g1 or g2) and g3", lambda a, b, c: (a or b) and c),
    ("g1 or (g2 and g3)", lambda a, b, c: a or (b and c)),
    ("(g1 and g2) or (g1 and g3)", lambda a, b, c: (a and b) or (a and c)),
    ("(g1 or g2) and (g1 or g3)", lambda a, b, c: (a or b) and (a or c)),
    ("g1 and g2 and g3", lambda a, b, c: a and b and c),
    ("g1 or g2 or g3", lambda a, b, c: a or b or c),
]
REDUCERS = [
    ("min", min), ("nanmin", min), ("max", max), ("nanmax", max),
    ("sum", sum), ("nansum", sum), ("mean", mean), ("nanmean", mean),
]


def truth_table_score(truth, values):
    """Find the highest expression cutoff that still satisfies a Boolean rule."""
    return max(level for level in set(values) if truth(*(v >= level for v in values)))


@pytest.mark.parametrize("rule_and_truth, expected", list(zip(BOOLEAN_RULES, [2, 10, 5, 5, 2, 5, 2, 10])),
                         ids=[r for r, _ in BOOLEAN_RULES])
def test_truth_table_oracle_matches_hand_calculated_anchor_answers(rule_and_truth, expected):
    # These fixed answers independently calibrate the helper used by the 512 cases.
    _, truth = rule_and_truth
    assert truth_table_score(truth, (2., 5., 10.)) == expected


def make_model(**rules):
    model = cobra.Model("mapping")
    reactions = []
    for rid, rule in rules.items():
        reaction = cobra.Reaction(rid)
        reaction.gene_reaction_rule = rule
        reactions.append(reaction)
    model.add_reactions(reactions)
    return model


@pytest.fixture(scope="module")
def boolean_models():
    return {rule: make_model(R=rule) for rule, _ in BOOLEAN_RULES}


@pytest.fixture(scope="module")
def arithmetic_model():
    return make_model(R="(g1 or g2) and (g3 or g4)")


@pytest.mark.parametrize("rule, expected", [
    ("g1 and (g2 or g3)", 2),
    ("(g1 or g2) and g3", 5),
    ("(g1 and g2) or (g3 and g4)", 7),
    ("g1 or (g2 and (g3 or g4))", 5),
])
def test_nested_rules_preserve_parentheses(rule, expected):
    data = GeneData({"g1": 2., "g2": 5., "g3": 10., "g4": 7.})
    data.align(make_model(R=rule))
    assert data.rxn_scores["R"] == expected


@pytest.mark.parametrize("gene_id", [
    "sensor", "candy", "surplus", "plus", "123", "gene-1", "a.b",
])
def test_gene_ids_are_not_operator_substrings(gene_id):
    mapper = RxnMapper(GeneData({gene_id: 7.}), make_model(R=gene_id))
    assert mapper.rxn_scores["R"] == 7


@pytest.mark.parametrize("and_op, or_op, expected", [
    ("nanmin", "nansum", 2),
    ("nansum", "nanmax", 12),
    ("nanmean", "nanmean", 4.75),
])
def test_custom_reducers_follow_the_gpr_tree(and_op, or_op, expected):
    mapper = RxnMapper(
        GeneData({"g1": 2., "g2": 5., "g3": 10.}),
        make_model(R="g1 and (g2 or g3)"),
        and_operation=and_op, or_operation=or_op,
    )
    assert mapper.rxn_scores["R"] == expected


@pytest.mark.parametrize("operation, expected", [("nanmin", 4), ("min", np.nan)])
def test_missing_genes_respect_nan_aware_reducers(operation, expected):
    mapper = RxnMapper(GeneData({"g1": 4.}), make_model(R="g1 and absent"),
                       and_operation=operation)
    assert mapper.rxn_scores["R"] == pytest.approx(expected, nan_ok=True)


def test_missing_rules_all_missing_and_threshold_boundary():
    mapper = RxnMapper(GeneData({"g1": 2., "g2": 3.}),
                       make_model(empty="", missing="unknown1 or unknown2",
                                  boundary="g1", present="g2"),
                       threshold=2, absent_value=-1)
    assert np.isnan(mapper.rxn_scores["empty"])
    assert np.isnan(mapper.rxn_scores["missing"])
    assert mapper.rxn_scores["boundary"] == -1
    assert mapper.rxn_scores["present"] == 3


def test_finite_missing_value_is_used_in_reduction():
    mapper = RxnMapper(GeneData({"g1": 4.}), make_model(R="g1 and unknown"),
                       missing_value=-2, threshold=-3)
    assert mapper.rxn_scores["R"] == -2


def test_empty_rule_keeps_missing_value_with_custom_plus_reducer():
    mapper = RxnMapper(GeneData({}), make_model(R=""),
                       missing_value=-2, threshold=-3, plus_operation="std")
    assert mapper.rxn_scores["R"] == -2


@pytest.mark.parametrize("rule, values, expected", [
    ("g1", {"g1": np.inf}, np.nan),
    ("g1 and g2", {"g1": np.inf, "g2": 4.}, 4),
    ("g1 or g2", {"g1": np.inf, "g2": 4.}, np.nan),
    ("g1 and g2", {"g1": -np.inf, "g2": 4.}, np.nan),
    ("g1 or g2", {"g1": -np.inf, "g2": 4.}, 4),
])
def test_nonfinite_scores_keep_existing_missing_policy(rule, values, expected):
    mapper = RxnMapper(GeneData(values, absent_expression=-np.inf), make_model(R=rule))
    assert mapper.rxn_scores["R"] == pytest.approx(expected, nan_ok=True)


@pytest.mark.parametrize("operation, expected", [("nansum", 6), ("nanmean", 3)])
def test_merged_reactions_combine_component_rules(operation, expected):
    model = make_model(first="g1 and (g2 or g3)", second="sensor", empty="")
    merged = MergedReaction("merged")
    merged.merged_rxns.update({r: 1 for r in model.reactions})
    model.add_reactions([merged])
    mapper = RxnMapper(GeneData({"g1": 2., "g2": 5., "g3": 10., "sensor": 4.}),
                       model, plus_operation=operation)
    assert mapper.rxn_scores["merged"] == expected


def test_partial_map_uses_new_data_preserves_options_and_updates_only_affected():
    model = make_model(changed="g1 or g2", untouched="g3", added="new_gene")
    mapper = RxnMapper(GeneData({"g1": 2., "g2": 5., "g3": 12.}), model,
                       threshold=10, absent_value=-1, or_operation="nansum")
    assert mapper.rxn_scores["changed"] == -1
    updated = GeneData({"g1": 7., "g2": 5., "g3": 99., "new_gene": 20.})
    mapper.partial_map(model, updated, ["g1", "new_gene"])
    assert mapper.rxn_scores == {"changed": 12, "untouched": 12, "added": 20}
    assert mapper.gene_data is updated.gene_data
    assert mapper.genes == updated.genes


def test_partial_map_accepts_generator_ids_and_explicit_option_overrides():
    model = make_model(R="g1", also="g1 and g2", other="g2")
    mapper = RxnMapper(GeneData({"g1": 2., "g2": 4.}), model, threshold=3)
    mapper.partial_map(model, GeneData({"g1": 5., "g2": 4.}),
                       (g for g in ["g1"]), threshold=6, absent_value=-2)
    assert mapper.rxn_scores == {"R": -2, "also": -2, "other": 4}


def test_partial_map_updates_merged_reactions_and_handles_removed_gene():
    model = make_model(first="g1", second="g2")
    merged = MergedReaction("merged")
    merged.merged_rxns.update({r: 1 for r in model.reactions})
    model.add_reactions([merged])
    mapper = RxnMapper(GeneData({"g1": 2., "g2": 4.}), model)
    mapper.partial_map(model, GeneData({"g2": 4.}), ["g1"])
    assert np.isnan(mapper.rxn_scores["first"])
    assert mapper.rxn_scores["second"] == 4
    assert mapper.rxn_scores["merged"] == 4


def test_mapping_observes_changed_gpr_without_stale_cache():
    model = make_model(R="g1")
    data = GeneData({"g1": 2., "g2": 7.})
    assert RxnMapper(data, model).rxn_scores["R"] == 2
    model.reactions.R.gene_reaction_rule = "g2"
    assert RxnMapper(data, model).rxn_scores["R"] == 7


@pytest.mark.parametrize("rule, truth", BOOLEAN_RULES, ids=[r for r, _ in BOOLEAN_RULES])
@pytest.mark.parametrize("values", list(product((0., 1., 2., 5.), repeat=3)))
def test_default_scores_match_exhaustive_boolean_truth_tables(rule, truth, values, boolean_models):
    expected = truth_table_score(truth, values)
    data = GeneData(dict(zip(("g1", "g2", "g3"), values)))
    mapper = RxnMapper(data, boolean_models[rule])
    assert mapper.rxn_scores["R"] == expected


@pytest.mark.parametrize("and_op, and_reference", REDUCERS, ids=[r for r, _ in REDUCERS])
@pytest.mark.parametrize("or_op, or_reference", REDUCERS, ids=[r for r, _ in REDUCERS])
@pytest.mark.parametrize("values", [
    (2., 5., 10., 7.), (0., 0., 0., 0.), (-3., 1., 4., -2.), (.125, .75, 1.25, 3.5),
])
def test_all_documented_reducer_pairs_against_python_arithmetic(
        and_op, and_reference, or_op, or_reference, values, arithmetic_model):
    a, b, c, d = values
    expected = and_reference([or_reference([a, b]), or_reference([c, d])])
    data = GeneData(dict(zip(("g1", "g2", "g3", "g4"), values)),
                    expression_threshold=-np.inf)
    mapper = RxnMapper(data, arithmetic_model,
                       and_operation=and_op, or_operation=or_op, threshold=-np.inf)
    assert mapper.rxn_scores["R"] == pytest.approx(expected)


# Explicit answers cover raw NaNs, bypassing GeneData's expression preprocessing.
@pytest.mark.parametrize("rule, values, options, expected", [
    ("g1", {}, {}, np.nan),
    ("g1", {"g1": 0.}, {}, 0),
    ("g1", {"g1": np.nan}, {}, np.nan),
    ("g1 and g2", {"g1": 4.}, {}, 4),
    ("g1 and g2", {"g1": 4.}, {"and_operation": "min"}, np.nan),
    ("g1 or g2", {"g1": 4.}, {}, 4),
    ("g1 or g2", {"g1": 4.}, {"or_operation": "max"}, np.nan),
    ("g1 or g2", {"g1": np.nan, "g2": 4.}, {"or_operation": "nansum"}, 4),
    ("g1 or g2", {"g1": np.nan, "g2": 4.}, {"or_operation": "sum"}, np.nan),
    ("g1 and g2", {"g1": np.nan, "g2": 4.}, {"and_operation": "nanmean"}, 4),
    ("g1 and g2", {"g1": np.nan, "g2": 4.}, {"and_operation": "mean"}, np.nan),
    ("g1 or g2", {}, {"or_operation": "nansum"}, np.nan),
    ("g1 and g2", {}, {"and_operation": "nansum"}, np.nan),
    ("g1 or g2", {"g1": np.nan, "g2": np.nan}, {"or_operation": "nanmean"}, np.nan),
    ("g1 and (g2 or g3)", {"g1": 2., "g3": 7.}, {}, 2),
    ("g1 and (g2 or g3)", {"g1": 2., "g3": 7.}, {"or_operation": "max"}, 2),
    ("g1 and (g2 or g3)", {"g1": 2., "g3": 7.},
     {"and_operation": "min", "or_operation": "max"}, np.nan),
    ("(g1 and g2) or g3", {"g1": 2., "g3": 7.}, {"and_operation": "min"}, 7),
    ("g1 and g2", {"g1": 4.}, {"missing_value": -2.}, -2),
    ("g1 or g2", {"g1": 4.}, {"missing_value": 6.}, 6),
    ("g1 or g2", {"g1": 4.}, {"missing_value": -2., "or_operation": "nansum"}, 2),
    ("g1 or g2", {"g1": np.inf, "g2": -np.inf}, {}, np.nan),
    ("g1 and g2", {"g1": np.inf, "g2": 4.}, {"and_operation": "nanmax"}, np.nan),
    ("g1 or g2", {"g1": np.inf, "g2": 4.}, {"missing_value": -2.}, -2),
    ("g1 and g2", {"g1": np.nan, "g2": np.inf}, {}, np.nan),
], ids=[
    "absent-gene", "measured-zero", "raw-nan", "partial-complex", "strict-complex",
    "partial-isozymes", "strict-isozymes", "nan-sum", "strict-sum", "nan-mean",
    "strict-mean", "all-missing-sum-or", "all-missing-sum-and", "all-nan-mean",
    "nested-missing", "nested-missing-ignored-at-parent", "nested-strict",
    "missing-branch-ignored", "negative-placeholder", "positive-placeholder",
    "placeholder-in-sum", "only-infinities", "nonfinite-maximum",
    "nonfinite-custom-placeholder", "no-finite-gene",
])
def test_missing_and_nonfinite_inputs_against_explicit_answers(rule, values, options, expected):
    data = SimpleNamespace(genes=list(values), gene_data=values)
    mapper = RxnMapper(data, make_model(R=rule), threshold=-np.inf, **options)
    assert mapper.rxn_scores["R"] == pytest.approx(expected, nan_ok=True)


@pytest.mark.parametrize("score, expected", [(-2., -7), (0., -7), (2., -7), (2.01, 2.01)])
def test_threshold_below_equal_and_above(score, expected):
    data = GeneData({"g1": score}, expression_threshold=-np.inf)
    mapper = RxnMapper(data, make_model(R="g1"), threshold=2, absent_value=-7)
    assert mapper.rxn_scores["R"] == expected


@pytest.mark.parametrize("missing, threshold, expected", [
    (np.nan, 0, np.nan), (-2., 0, -7), (-2., -3, -2), (4., 4, -7),
])
def test_threshold_applies_to_finite_missing_placeholders(missing, threshold, expected):
    mapper = RxnMapper(GeneData({}), make_model(empty="", unknown="missing_gene"),
                       missing_value=missing, threshold=threshold, absent_value=-7)
    for score in mapper.rxn_scores.values():
        assert score == pytest.approx(expected, nan_ok=True)


@pytest.mark.parametrize("gene_id", [
    "floor", "standard", "plus_one", "ORF1", "1abc", "gene:1", "gene/1", "gene=1", "class",
])
def test_unusual_identifiers_inside_compound_rules(gene_id):
    expected = min(3, max(7, 5))
    mapper = RxnMapper(GeneData({"g1": 3., gene_id: 7., "g2": 5.}),
                       make_model(R=f"g1 and ({gene_id} or g2)"))
    assert mapper.rxn_scores["R"] == expected


@pytest.mark.parametrize("rule", [
    "g1 and g1", "g1 or g1", "((g1))", "g1 and (g1 or g1)",
    "g1 OR g1", "g1 AND g1", "g1 & g1", "g1 | g1",
])
def test_idempotence_and_cobra_supported_operator_spellings(rule):
    mapper = RxnMapper(GeneData({"g1": 3.}), make_model(R=rule))
    assert mapper.rxn_scores["R"] == 3


@pytest.mark.parametrize("gene_ids", [[], ["unknown"], ["g1"], ["g1", "g1"], ["g1", "g3"]])
@pytest.mark.parametrize("container", [list, tuple, set, iter])
def test_constructor_subset_matches_explicit_reaction_membership(gene_ids, container):
    rules = {"first": "g1", "second": "g2 and g3", "third": "g1 or g2", "empty": ""}
    members = {"first": {"g1"}, "second": {"g2", "g3"}, "third": {"g1", "g2"}, "empty": set()}
    answers = {"first": 2, "second": 5, "third": 5}
    expected = {rid: answers[rid] for rid in members if members[rid].intersection(gene_ids)}
    mapper = RxnMapper(GeneData({"g1": 2., "g2": 5., "g3": 7.}),
                       make_model(**rules), gene_ids=container(gene_ids))
    assert mapper.rxn_scores == expected


@pytest.mark.parametrize("container", [list, tuple, set, iter])
def test_repeated_partial_updates_match_independent_truth_tables(container):
    rules = {f"R{i}": rule for i, (rule, _) in enumerate(BOOLEAN_RULES)}
    current = {"g1": 2., "g2": 5., "g3": 7.}
    expected = {rid: truth_table_score(truth, tuple(current.values()))
                for rid, (_, truth) in zip(rules, BOOLEAN_RULES)}
    model = make_model(**rules)
    mapper = RxnMapper(GeneData(current), model)
    assert mapper.rxn_scores == expected
    for changes in ({"g1": 11.}, {"g2": 0.}, {"g1": 1., "g3": 3.}):
        current.update(changes)
        # Every rule in this table uses every gene; all are affected by each update.
        expected = {rid: truth_table_score(truth, tuple(current.values()))
                    for rid, (_, truth) in zip(rules, BOOLEAN_RULES)}
        mapper.partial_map(model, GeneData(current), container(changes))
        assert mapper.rxn_scores == expected


@pytest.mark.parametrize("gene_ids", [[], ["unknown"]])
def test_partial_update_with_no_affected_reactions_preserves_scores(gene_ids):
    model = make_model(R="g1", empty="")
    mapper = RxnMapper(GeneData({"g1": 2.}), model)
    mapper.partial_map(model, GeneData({"g1": 99.}), iter(gene_ids))
    assert mapper.rxn_scores["R"] == 2
    assert np.isnan(mapper.rxn_scores["empty"])


def test_partial_update_retains_custom_reducers_and_threshold_across_updates():
    model = make_model(R="(g1 or g2) and g3", untouched="g4")
    mapper = RxnMapper(GeneData({"g1": 1., "g2": 3., "g3": 4., "g4": 20.}), model,
                       and_operation="nansum", or_operation="nanmean", threshold=9, absent_value=-1)
    assert mapper.rxn_scores["R"] == -1
    for g1, expected in [(9., 10.), (3., -1), (13., 12.)]:
        # (g1 + 3) / 2 + 4, thresholded at <= 9.
        mapper.partial_map(model, GeneData({"g1": g1, "g2": 3., "g3": 4., "g4": 20.}), ["g1"])
        assert mapper.rxn_scores == {"R": expected, "untouched": 20}


def test_partial_update_replaces_constructor_subset_and_override_is_temporary():
    model = make_model(first="g1", second="g2")
    mapper = RxnMapper(GeneData({"g1": 2., "g2": 5.}), model, gene_ids=["g1"], threshold=3)
    assert mapper.rxn_scores == {"first": 0}
    updated = GeneData({"g1": 2., "g2": 7.})
    mapper.partial_map(model, updated, ["g2"], threshold=8, absent_value=-1)
    assert mapper.rxn_scores == {"first": 0, "second": -1}
    mapper.partial_map(model, updated, ["g2"])
    assert mapper.rxn_scores == {"first": 0, "second": 7}


@pytest.mark.parametrize("operation, reference", REDUCERS, ids=[r for r, _ in REDUCERS])
def test_actual_repeated_merging_against_component_arithmetic(operation, reference):
    a, b, c, d = [cobra.Metabolite(name) for name in "abcd"]
    model = make_model(first="g1 and g2", second="g3 or g4", third="sensor")
    model.reactions.first.add_metabolites({a: -1, b: 1})
    model.reactions.second.add_metabolites({b: -2, c: 1})
    model.reactions.third.add_metabolites({c: -1, d: 1})
    first_merge = MergedReaction("first_merge")
    first_merge.merge_two_rxns([model.reactions.first, model.reactions.second], b)
    model.add_reactions([first_merge])
    merged = MergedReaction("merged")
    merged.merge_two_rxns([first_merge, model.reactions.third], c)
    empty = cobra.Reaction("no_genes")
    merged.merged_rxns[empty] = 1
    model.add_reactions([merged])
    # Stoichiometric coefficients differ, but GPR component scores are unweighted.
    expected = reference([min(2, 5), max(3, 7), 4])
    mapper = RxnMapper(GeneData({"g1": 2., "g2": 5., "g3": 3., "g4": 7., "sensor": 4.}),
                       model, plus_operation=operation)
    assert mapper.rxn_scores["merged"] == pytest.approx(expected)


@pytest.mark.parametrize("plus_op, expected", [
    ("nansum", 7), ("nanmin", 7), ("nanmax", 7), ("nanmean", 7),
    ("sum", np.nan), ("min", np.nan), ("max", np.nan), ("mean", np.nan),
])
def test_merged_missing_component_reductions(plus_op, expected):
    model = make_model(unknown="g1 and g2", known="g3", empty="")
    merged = MergedReaction("merged")
    merged.merged_rxns.update({reaction: 1 for reaction in model.reactions})
    model.add_reactions([merged])
    mapper = RxnMapper(GeneData({"g3": 7.}), model, plus_operation=plus_op)
    assert mapper.rxn_scores["merged"] == pytest.approx(expected, nan_ok=True)


@pytest.mark.parametrize("rules", [{}, {"missing": "g1"}, {"empty": ""}])
def test_empty_or_all_missing_merged_reaction_stays_missing(rules):
    model = make_model(**rules)
    merged = MergedReaction("merged")
    merged.merged_rxns.update({reaction: 1 for reaction in model.reactions})
    model.add_reactions([merged])
    mapper = RxnMapper(GeneData({}), model)
    assert np.isnan(mapper.rxn_scores["merged"])


def test_finite_missing_component_counts_but_empty_rule_does_not():
    model = make_model(known="g1", unknown="g2", empty="")
    merged = MergedReaction("merged")
    merged.merged_rxns.update({reaction: 1 for reaction in model.reactions})
    model.add_reactions([merged])
    expected = (4 + 0) / 2
    mapper = RxnMapper(GeneData({"g1": 4.}), model,
                       missing_value=0, threshold=-1, plus_operation="nanmean")
    assert mapper.rxn_scores["merged"] == expected


def test_empty_model_has_no_scores():
    assert RxnMapper(GeneData({"g1": 2.}), make_model()).rxn_scores == {}


@pytest.mark.parametrize("option", ["and_operation", "or_operation", "plus_operation"])
def test_invalid_reducer_name_raises_clear_attribute_error(option):
    with pytest.raises(AttributeError, match="not_a_reducer"):
        RxnMapper(GeneData({"g1": 2.}), make_model(R="g1"), **{option: "not_a_reducer"})


def test_corrupted_gpr_fails_loudly_instead_of_creating_a_score():
    model = make_model(R="g1")
    model.reactions.R.gpr.body = Constant(value=True)
    with pytest.raises(TypeError, match="Unsupported GPR node: Constant"):
        RxnMapper(GeneData({"g1": 2.}), model)


def test_overflowing_sum_returns_missing_instead_of_infinite_activity():
    data = GeneData({"g1": 1e308, "g2": 1e308})
    # The mathematical sum exceeds the finite float range: the score is unknown.
    with np.errstate(over="ignore"):
        mapper = RxnMapper(data, make_model(R="g1 or g2"), or_operation="nansum")
    assert np.isnan(mapper.rxn_scores["R"])


def test_transform_and_threshold_apply_after_raw_gene_aggregation():
    expected_raw = 1 + 9
    expected_transformed = sqrt(expected_raw)
    data = GeneData({"g1": 1., "g2": 9.}, data_transform="sqrt")
    data.align(make_model(R="g1 or g2"), or_operation="nansum", threshold=9.5, absent_value=-1)
    assert data.rxn_mapper.rxn_scores["R"] == expected_raw
    assert data.rxn_scores["R"] == pytest.approx(expected_transformed)


def test_integer_measurement_ids_align_with_cobra_string_ids():
    mapper = RxnMapper(GeneData({123: 7.}), make_model(R="123"))
    assert mapper.rxn_scores["R"] == 7


def test_multiple_samples_through_model_api_have_independent_correct_scores():
    # Explicit calculations distinguish parentheses, sample reuse, and transform order.
    samples = pd.DataFrame({"A": [2., 5., 10.], "B": [8., 3., 4.]}, index=["g1", "g2", "g3"])
    expected = {"sample_A": sqrt(min(2, 5 + 10)), "sample_B": sqrt(min(8, 3 + 4))}
    model = Model(model=make_model(R="g1 and (g2 or g3)"))
    model.add_gene_data("sample", samples, data_kwargs={"data_transform": "sqrt"}, or_operation="nansum")
    for sample, answer in expected.items():
        assert model.gene_data[sample].rxn_scores["R"] == pytest.approx(answer)


@pytest.mark.parametrize("model_name", ["textbook", "iJO1366"])
def test_real_model_scores_match_boolean_knockout_sweeps(model_name):
    # COBRA ships these models; no download or network access is needed.
    path = Path(cobra.__file__).parent / "data" / f"{model_name}.xml.gz"
    # Use a fresh process for native SBML loading under Windows coverage.
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; from cobra.io import read_sbml_model, to_json; "
         "print(to_json(read_sbml_model(sys.argv[1])))", str(path)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    model = cobra.io.from_json(result.stdout)
    values = {gene.id: float(index % 7 + 1) for index, gene in enumerate(model.genes)}
    knockouts = {level: {gene for gene, value in values.items() if value < level} for level in range(1, 8)}
    expected = {}
    for reaction in model.reactions:
        if not reaction.genes:
            expected[reaction.id] = np.nan
        else:
            expected[reaction.id] = max(level for level in range(1, 8) if reaction.gpr.eval(knockouts[level]))
    mapper = RxnMapper(GeneData(values), model)
    assert mapper.rxn_scores == pytest.approx(expected, nan_ok=True)
