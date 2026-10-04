import pandas as pd
import numpy as np
import cobra
import pytest

from pipeGEM.core import Model
from pipeGEM.data import GeneData
from pipeGEM.analysis import DataAggregation


@pytest.mark.parametrize("integrator", ["FASTCORE", "SWIFTCORE", "SPOT"])
def test_advertised_integrators_use_public_model_api(trivial_linear_model, integrator):
    mod = Model(model=trivial_linear_model)
    mod.add_gene_data("sample", pd.Series({"g1": 10., "g2": 1., "g3": 1., "g4": 1.}))
    original_bounds = {r.id: r.bounds for r in mod.reactions}
    result = mod.integrate_gene_data(
        "sample", integrator=integrator,
        predefined_threshold={"exp_th": 5., "non_exp_th": 0.},
        protected_rxns=["BIOMASS"],
    )
    if integrator == "SPOT":
        assert result.flux_result.loc["BIOMASS", "fluxes"] >= 1.
    else:
        assert {"R1", "BIOMASS"} <= {r.id for r in result.result_model.reactions}
        assert result.result_model.slim_optimize() == pytest.approx(10.)
        assert set(result.kept_rxn_ids) == {r.id for r in result.result_model.reactions}
    assert {r.id: r.bounds for r in mod.reactions} == original_bounds


@pytest.mark.parametrize("integrator", ["FASTCORE", "SWIFTCORE"])
def test_core_integrators_validate_explicit_reaction_ids(trivial_linear_model, integrator):
    mod = Model(model=trivial_linear_model)
    mod.add_gene_data("sample", pd.Series({"g1": 10., "g2": 1., "g3": 1., "g4": 1.}))
    with pytest.raises(ValueError, match="Unknown core reaction IDs.*missing"):
        mod.integrate_gene_data("sample", integrator=integrator, C=["missing"])


@pytest.mark.parametrize("integrator", ["FASTCORE", "SWIFTCORE"])
@pytest.mark.parametrize("explicit_core", [False, True])
def test_core_integrators_select_and_persist_core(trivial_linear_model, integrator, tmp_path, explicit_core):
    mod = Model(model=trivial_linear_model)
    mod.add_gene_data("sample", pd.Series({"g1": 10., "g2": 1., "g3": 1., "g4": 1.}))
    weights = np.ones(len(mod.reactions), dtype=int)
    kwargs = {"weights": weights} if integrator == "SWIFTCORE" else {"nonP": np.array(["R_transport"])}
    result = mod.integrate_gene_data("sample", integrator=integrator, C=["R2"] if explicit_core else None,
                                     protected_rxns=["BIOMASS"], **kwargs)
    assert {"R2" if explicit_core else "R1", "BIOMASS"} <= {r.id for r in result.result_model.reactions}
    result.save(tmp_path / integrator)
    loaded = type(result).load(tmp_path / integrator)
    assert loaded.result_model.slim_optimize() == pytest.approx(10.)
    np.testing.assert_array_equal(weights, 1)


def test_swiftcore_zero_lower_bounds(trivial_linear_model):
    # Express uptake in the forward direction so every lower bound is zero.
    uptake = trivial_linear_model.reactions.EX_A
    uptake.add_metabolites({met: -2 * coef for met, coef in uptake.metabolites.items()})
    uptake.bounds = (0, 10)
    mod = Model(model=trivial_linear_model)
    mod.add_gene_data("sample", pd.Series({"g1": 10., "g2": 1., "g3": 1., "g4": 1.}))
    with np.errstate(divide="raise", invalid="raise"):
        result = mod.integrate_gene_data("sample", integrator="SWIFTCORE", C=["R1"], protected_rxns=["BIOMASS"])
    assert "R1" in result.kept_rxn_ids
    assert result.result_model.slim_optimize() == pytest.approx(trivial_linear_model.slim_optimize())


def test_init_model(ecoli_core):
    mod = Model(model=ecoli_core, name_tag="ecoli")
    assert mod is not None
    assert len(mod.reactions) == len(ecoli_core.reactions)


def test_model_flux_analysis(ecoli_core):
    mod = Model(model=ecoli_core, name_tag="ecoli")
    result = mod.do_flux_analysis(method="pFBA", solver="glpk")
    assert isinstance(result.flux_df, pd.DataFrame)


def test_model_add_data(ecoli_core, ecoli_core_data):
    pmod = Model(model=ecoli_core, name_tag="ecoli")
    data_name = "sample_0"
    gene_data = GeneData(data=ecoli_core_data[data_name], data_transform=lambda x: np.log2(x), absent_expression=-np.inf)
    pmod.add_gene_data(data_name, gene_data)
    assert isinstance(pmod.gene_data[data_name].rxn_scores, dict)


def test_model_aggregate_data(ecoli_core, ecoli_core_data):
    group_info = pd.DataFrame({"grp": {data_name: i % 5
                                       for i, (data_name, data) in enumerate(ecoli_core_data.items())}
                               })
    pmod = Model(model=ecoli_core, name_tag="ecoli", gene_data_factor_df=group_info)
    for d_name, data in ecoli_core_data.items():
        gene_data = GeneData(data=data,
                             data_transform=lambda x: np.log2(x), absent_expression=-np.inf)
        pmod.add_gene_data(d_name, gene_data)
    assert isinstance(pmod.gene_data["sample_0"].rxn_scores, dict)
    agg_data = pmod.aggregate_gene_data()
    th = agg_data.find_local_threshold(group_name="grp", p=50)
    assert th is not None
    assert hasattr(th, "exp_ths")


def test_check_model_scale_geometric_mean(ecoli_core):
    mod = Model(model=ecoli_core, name_tag="ecoli")
    mod.reactions[0].add_metabolites({k: v * 99999 for k, v in mod.reactions[0].metabolites.items()})
    rescale_result = mod.check_model_scale(n_iter=5)
    assert rescale_result is not None
    assert hasattr(rescale_result, "decimals")
    assert hasattr(rescale_result, "diff_A")


def test_check_model_scale_arithmetic(ecoli_core):
    mod = Model(model=ecoli_core, name_tag="ecoli")
    # Mess up stoichiometry to test rescaling
    mod.reactions[0].add_metabolites({k: v * 99999 for k, v in mod.reactions[0].metabolites.items()})
    rescale_result = mod.check_model_scale(method="arithmetic", n_iter=5)
    assert rescale_result is not None
    assert hasattr(rescale_result, "diff_A")
    assert hasattr(rescale_result, "rescaled_model")

    reversed_rescaled = rescale_result.reverse_scaling(rescale_result.rescaled_model)
    assert abs(reversed_rescaled.reactions[0].lower_bound - mod.reactions[0].lower_bound) < 1e-4, \
        reversed_rescaled.reactions[0].bounds
    assert abs(reversed_rescaled.reactions[0].upper_bound - mod.reactions[0].upper_bound) < 1e-4, \
        reversed_rescaled.reactions[0].bounds
