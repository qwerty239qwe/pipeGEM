import pandas as pd
import numpy as np
import cobra
import pytest

from pipeGEM.core import Model
from pipeGEM.data import GeneData
from pipeGEM.analysis import DataAggregation


def test_init_model(ecoli_core):
    return Model(model=ecoli_core, name_tag="ecoli")


def test_model_flux_analysis(ecoli_core):
    mod = Model(model=ecoli_core, name_tag="ecoli")
    result = mod.do_flux_analysis(method="pFBA", solver="glpk")
    assert isinstance(result.flux_df, pd.DataFrame)


def test_add_data_model(ecoli_core, ecoli_core_data):
    pmod = Model(model=ecoli_core, name_tag="ecoli")
    data_name = "sample_0"
    gene_data = GeneData(data=ecoli_core_data[data_name], data_transform=lambda x: np.log2(x), absent_expression=-np.inf)
    pmod.add_gene_data(data_name, gene_data)
    assert isinstance(pmod.gene_data[data_name].rxn_scores, dict)


def test_check_model_scale_geometric_mean(ecoli_core):
    mod = Model(model=ecoli_core, name_tag="ecoli")
    mod.reactions[0].add_metabolites({k: v * 99999 for k, v in mod.reactions[0].metabolites.items()})
    rescale_result = mod.check_model_scale(n_iter=5)
    print(abs(rescale_result.rescaled_A).max())
    print(abs(rescale_result.diff_A).max())


def test_check_model_scale_arithmetic(ecoli_core):
    mod = Model(model=ecoli_core, name_tag="ecoli")
    print(mod.optimize())
    print(mod.reactions.BIOMASS_Ecoli_core_w_GAM)

    # let's mess this around
    mod.reactions[0].add_metabolites({k: v * 99999 for k, v in mod.reactions[0].metabolites.items()})
    print(mod.reactions[0], mod.reactions[0].lower_bound, mod.reactions[0].upper_bound)
    # for r in mod.metabolites[1].reactions:
    #     print(f"add 99999 {mod.metabolites[0].id} to {r.id}")
    #     r.add_metabolites({mod.metabolites[0]: 99999})
    rescale_result = mod.check_model_scale(method="arithmetic", n_iter=5)
    print(abs(rescale_result.rescaled_A).max())
    print(abs(rescale_result.diff_A).max())
    print(mod.optimize())

    print(rescale_result.rescaled_model.reactions[0], rescale_result.rescaled_model.reactions[0].lower_bound, rescale_result.rescaled_model.reactions[0].upper_bound)
    print(rescale_result.rescaled_model.optimize())
    print(rescale_result.rescaled_model.reactions.BIOMASS_Ecoli_core_w_GAM)