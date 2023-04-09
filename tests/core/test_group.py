import pandas as pd
import numpy as np
import cobra
import pytest

from pipeGEM.core import Model
from pipeGEM.core import Group
from pipeGEM.data import GeneData
from pipeGEM.analysis import DataAggregation


@pytest.fixture(scope="module")
def group(ecoli_core):
    return Group({"ecoli_g1": {"m111": ecoli_core, "m112": ecoli_core, "m12": ecoli_core},
                    "ecoli_g2": {"m21": ecoli_core, "m22": ecoli_core},
                    "ecoli_g3": {"m3": ecoli_core}}, name_tag="G2",
                    treatment={"m111": "a", "m112": "b"})


def test_init_group(ecoli_core):
    m1 = ecoli_core
    g = Group({"ecoli": m1, "ecoli_2": m1.copy()}, name_tag="G1")
    assert m1 == g["ecoli"].cobra_model
    assert isinstance(g["ecoli"], Model)


def test_init_group2(ecoli_core):
    m1 = ecoli_core
    g2 = Group({"ecoli_g1": {"e11": m1, "e12": m1},
                "ecoli_g2": {"e21": m1, "e22": m1}}, name_tag="G2")
    print(g2._group_annotation)
    print(g2.annotation)
    assert g2["e11"].name_tag == 'e11'
    assert g2.annotation.loc["e11", "group_name"] == "ecoli_g1"


def test_group_get_info(group, ecoli_core):
    assert len(group.reaction_ids) == len(ecoli_core.reactions)
    assert len(group.metabolite_ids) == len(ecoli_core.metabolites)
    assert len(group.gene_ids) == len(ecoli_core.genes)
    assert isinstance(group.get_info(), pd.DataFrame), group.get_info()


def test_group_get_info_2(ecoli_core):
    m1 = ecoli_core
    g2 = Group({"ecoli_g1": {"m111": m1, "m112": m1, "m12": m1},
                "ecoli_g2": {"m21": m1, "m22": m1},
                "ecoli_g3": {"m3": m1}}, name_tag="G2",
                treatment={"m111": "a", "m112": "b"})
    print(g2.get_info(features=["n_rxns", "n_mets", "n_genes", "treatment"]))
    assert g2.get_info(features=["n_rxns", "n_mets", "n_genes", "treatment"]).loc["m111", "treatment"] == "a"
    assert g2.get_info(features=["n_rxns", "n_mets", "n_genes", "treatment"]).loc["m3", "treatment"] is None


def test_group_get_info_3(ecoli_core):
    m1 = ecoli_core
    g3 = Group({"ecoli_g1": {"m111": m1, "m112": m1, "m12": m1},
                "ecoli_g2": {"m21": m1, "m22": m1},
                "ecoli_g3": {"m3": m1}}, name_tag="G2",
               treatment={"a": ["m3", "m111", "m21"], "b": ["m112", "m12"]})
    assert g3.get_info(features=["n_rxns", "n_mets", "n_genes", "treatment"]).loc["m111", "treatment"] == "a"
    assert g3.get_info(features=["n_rxns", "n_mets", "n_genes", "treatment"]).loc["m112", "treatment"] == "b"
    assert g3.get_info(features=["n_rxns", "n_mets", "n_genes", "treatment"]).loc["m22", "treatment"] is None


def test_group_get_flux(group):
    fba_result = group.do_flux_analysis(method="FBA", solver="glpk")
    sampling_result = group.do_flux_analysis(method="sampling", solver="glpk", n=10)
    print(sampling_result.result)
    assert isinstance(fba_result.result, pd.DataFrame)
    assert isinstance(sampling_result.result, dict)


def test_group_get_items(group):
    assert group["m111"].name_tag == "m111"


def test_group_set_items(group, ecoli_core):
    m1 = ecoli_core
    assert len(group)==6
    assert isinstance(group["m111"], Model)
    group["m4"] = m1
    assert len(group)==7
    assert "m4" in group


def test_aggregate_models(group):
    ag_grp_dic = group.aggregate_models("treatment")
    assert isinstance(ag_grp_dic, dict)


def test_compare_sim(group):
    sim_comp = group.compare(method="jaccard")
    sim_comp.plot(dpi=150)


def test_compare_num(ecoli_core):
    m1 = ecoli_core
    g = Group(group={"ecoli_g1": {"e11": m1, "e12": m1, "e13": m1},
                     "ecoli_g2": {"e21": m1, "e22": m1}},
              name_tag="G2")
    num_comp = g.compare(models=["e11", "e21", "e22"], method="num")
    print(num_comp.result)
    print(num_comp.name_order)
    num_comp.plot(dpi=150)

    num_comp = g.compare(tags=None, method="num")
    print(num_comp.result)
    print(num_comp.name_order)
    num_comp.plot(dpi=150, name_order=[gi.name_tag for gi in g])


def test_compare_PCA(ecoli_core):
    m1 = ecoli_core
    g = Group(group={"ecoli_g1": {"e11": m1, "e12": m1, "e13": m1},
                     "ecoli_g2": {"e21": m1, "e22": m1}, "a": m1},
              name_tag="G2")
    num_comp = g.compare(tags=None, compare_models=True, use="PCA")
    print(num_comp.result)
    num_comp.plot(dpi=150)

    num_comp = g.compare(tags=None, compare_models=False, use="PCA")
    print(num_comp.result)
    num_comp.plot(dpi=150)

# TODO: test_add_tasks

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