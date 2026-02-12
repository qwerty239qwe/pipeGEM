import pandas as pd
import numpy as np
import cobra
import pytest

from pipeGEM.core import Model
from pipeGEM.data import GeneData
from pipeGEM.analysis import DataAggregation


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