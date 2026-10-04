import pandas as pd
import pytest

from pipeGEM import Model


@pytest.mark.parametrize("method", ["FBA", "pFBA"])
def test_single_model_flux_round_trip(tmp_path, trivial_linear_model, method):
    result = Model(model=trivial_linear_model).do_flux_analysis(method, solver="glpk")
    result.save(tmp_path / method)
    loaded = type(result).load(tmp_path / method)
    pd.testing.assert_frame_equal(loaded.flux_df, result.flux_df)
    assert loaded.solution.status == result.solution.status
    assert loaded.solution.objective_value == pytest.approx(result.solution.objective_value)
    for field in ("fluxes", "reduced_costs", "shadow_prices"):
        pd.testing.assert_series_equal(getattr(loaded.solution, field), getattr(result.solution, field))


def test_categorical_flux_metadata_round_trip(tmp_path, trivial_linear_model):
    result = Model(model=trivial_linear_model).do_flux_analysis("FBA", solver="glpk")
    result.add_categorical("sample", col_name="model")
    result.save(tmp_path / "flux")
    loaded = type(result).load(tmp_path / "flux")
    loaded.add_categorical("control", col_name="condition")
    assert loaded.log["categorical"] == {"model", "condition"}
    aggregated = type(result).aggregate([loaded, loaded], "concat")
    assert len(aggregated.flux_df) == 2 * len(loaded.flux_df)


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
