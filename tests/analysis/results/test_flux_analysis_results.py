import pandas as pd


def test_sampling_get_item(sampling_result, ecoli_core):
    rxn_ids = [r.id for r in ecoli_core.reactions]
    subset_1 = sampling_result[rxn_ids[0]]
    assert rxn_ids[0] in subset_1.flux_df.columns


def test_sampling_get_multiple_item(sampling_result, ecoli_core):
    rxn_ids = [r.id for r in ecoli_core.reactions]
    sel_rxns = rxn_ids[3: 10]
    subset_2 = sampling_result[sel_rxns]
    assert all([r in subset_2.flux_df.columns for r in sel_rxns])
    assert all([r not in subset_2.flux_df.columns for r in rxn_ids if r not in sel_rxns])


def test_sampling_add(sampling_result, ecoli_core):
    rxn_ids = [r.id for r in ecoli_core.reactions]
    sel_rxns = rxn_ids[3: 10]
    subset_1 = sampling_result[sel_rxns]
    subset_2 = sampling_result[sel_rxns] + 10

    sum_subset = subset_1 + subset_2
    assert all([(sum_subset.flux_df[r] - (sampling_result.flux_df[r] * 2 + 10)).sum() < 1e-6 for r in sel_rxns])


def test_sampling_operate_sum(sampling_result, ecoli_core):
    rxn_ids = [r.id for r in ecoli_core.reactions]
    sel_rxns = rxn_ids[3: 10]

    sampling_result.operate("ans=" + "+".join(rxn_ids[3: 10]))
    ans = sampling_result.flux_df["ans"]

    assert isinstance(ans, pd.Series)
    assert (ans - sampling_result.flux_df[sel_rxns].sum(axis=1)).sum() < 1e-6