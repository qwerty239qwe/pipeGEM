import pandas as pd
import numpy as np
import pytest
from pipeGEM.data.synthesis import get_syn_gene_data
from pipeGEM.core import Model
from pipeGEM.core import Group
from pipeGEM.data import GeneData
from pipeGEM.analysis import DataAggregation


@pytest.fixture(scope="session")
def ecoli_core_data(ecoli_core):
    return get_syn_gene_data(ecoli_core, n_sample=100)


def test_add_data_model(ecoli_core, ecoli_core_data):
    pmod = Model(model=ecoli_core, name_tag="ecoli")
    data_name = "sample_0"
    gene_data = GeneData(data=ecoli_core_data[data_name], data_transform=lambda x: np.log2(x), absent_expression=-np.inf)
    pmod.add_gene_data(data_name, gene_data)
    assert pmod.gene_data[data_name].rxn_scores


def test_add_data_group(ecoli_core, ecoli_core_data):
    n_model = 4
    pmods = [Model(model=ecoli_core, name_tag=f"ecoli_{i}") for i in range(n_model)]
    for i in range(n_model):
        data_name = f"sample_{i}"
        gene_data = GeneData(data=ecoli_core_data[data_name], data_transform=lambda x: np.log2(x),
                             absent_expression=-np.inf)
        pmods[i].add_gene_data(data_name, gene_data)
    grp = Group({f"ecoli_{i}": m for i, m in enumerate(pmods)}, name_tag="group1")
    assert isinstance(grp.gene_data, DataAggregation)