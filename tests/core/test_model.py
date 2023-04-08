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
    assert isinstance(result.result, pd.DataFrame)


def test_add_data_model(ecoli_core, ecoli_core_data):
    pmod = Model(model=ecoli_core, name_tag="ecoli")
    data_name = "sample_0"
    gene_data = GeneData(data=ecoli_core_data[data_name], data_transform=lambda x: np.log2(x), absent_expression=-np.inf)
    pmod.add_gene_data(data_name, gene_data)
    assert pmod.gene_data[data_name].rxn_scores