import pandas as pd
import numpy as np
import pytest
from pipeGEM.data.synthesis import get_syn_gene_data
from pipeGEM.core import Model
from pipeGEM.core import Group
from pipeGEM.data import GeneData
from pipeGEM.analysis import DataAggregation, PercentileThresholdAnalysis, rFASTCORMICSThresholdAnalysis


def test_add_data_model(ecoli_core, ecoli_core_data):
    pmod = Model(model=ecoli_core, name_tag="ecoli")
    data_name = "sample_0"
    gene_data = GeneData(data=ecoli_core_data[data_name],
                         data_transform=lambda x: np.log2(x),
                         absent_expression=-np.inf)
    pmod.add_gene_data(data_name, gene_data)
    assert pmod.gene_data[data_name].rxn_scores


def test_add_data_model_human(Human_GEM, Human_GEM_data):
    pmod = Model(model=Human_GEM, name_tag="human")
    data_name = "sample_0"
    gene_data = GeneData(data=Human_GEM_data[data_name],
                         data_transform=lambda x: np.log2(x),
                         absent_expression=-np.inf)
    pmod.add_gene_data(data_name, gene_data)
    assert isinstance(pmod.gene_data[data_name].rxn_scores, dict)


def test_find_percentile_threshold(ecoli_core_data):
    data_name = "sample_0"
    gene_data = GeneData(data=ecoli_core_data[data_name],
                         data_transform=lambda x: np.log2(x),
                         absent_expression=-np.inf)
    p10 = gene_data.get_threshold(name="percentile", p=10)
    p50 = gene_data.get_threshold(name="percentile", p=50)
    p90 = gene_data.get_threshold(name="percentile", p=90)
    assert isinstance(p10, PercentileThresholdAnalysis)
    assert isinstance(p50, PercentileThresholdAnalysis)
    assert isinstance(p90, PercentileThresholdAnalysis)
    assert p10.exp_th <= p50.exp_th <= p90.exp_th


def test_find_rFASTCORMICS_threshold(ecoli_core_data):
    data_name = "sample_0"
    gene_data = GeneData(data=ecoli_core_data[data_name],
                         data_transform=lambda x: np.log2(x),
                         absent_expression=-np.inf)
    rth = gene_data.get_threshold(name="rFASTCORMICS")
    assert isinstance(rth, rFASTCORMICSThresholdAnalysis)
    assert rth.exp_th > rth.non_exp_th
    rth.save("./rFASTCORMICS")