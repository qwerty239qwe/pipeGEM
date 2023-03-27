import pandas as pd
import numpy as np
import pytest
from pipeGEM.data.synthesis import get_syn_gene_data
from pipeGEM.core import Model
from pipeGEM.core import Group
from pipeGEM.data import GeneData
from pipeGEM.analysis import DataAggregation


def test_add_data_model(ecoli_core, ecoli_core_data):
    pmod = Model(model=ecoli_core, name_tag="ecoli")
    data_name = "sample_0"
    gene_data = GeneData(data=ecoli_core_data[data_name], data_transform=lambda x: np.log2(x), absent_expression=-np.inf)
    pmod.add_gene_data(data_name, gene_data)
    assert pmod.gene_data[data_name].rxn_scores


def test_add_data_model_human(Human_GEM, Human_GEM_data):
    pmod = Model(model=Human_GEM, name_tag="human")
    data_name = "sample_0"
    gene_data = GeneData(data=Human_GEM_data[data_name],
                         data_transform=lambda x: np.log2(x), absent_expression=-np.inf)
    pmod.add_gene_data(data_name, gene_data)
    assert pmod.gene_data[data_name].rxn_scores