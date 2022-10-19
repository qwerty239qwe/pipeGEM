import pandas as pd
import cobra

from pipeGEM.core import Model
from pipeGEM.core import Group


def test_init_model(ecoli_core):
    return Model(model=ecoli_core, name_tag="ecoli")


def test_model_flux_analysis(ecoli_core):
    mod = Model(model=ecoli_core, name_tag="ecoli")
    result = mod.do_flux_analysis(method="pFBA", solver="glpk")
    assert isinstance(result.result, pd.DataFrame)


def test_get_group(ecoli_core):
    m1 = ecoli_core
    g = Group({"ecoli": m1, "ecoli_2": m1.copy()}, name_tag="G1")
    assert m1 == g["ecoli"].cobra_model
    assert isinstance(g["ecoli"], Model)
    assert g._lvl == 2


def test_get_group2(ecoli_core):
    m1 = ecoli_core
    g2 = Group({"ecoli_g1": {"e11": m1, "e12": m1}, "ecoli_g2": {"e21": m1, "e22": m1}, "a": m1}, name_tag="G2")
    assert g2.iget(0).name_tag == 'ecoli_g1'
    print(g2.iget(0)._group)
    assert g2.iget(0).iget(0).name_tag == "e11", g2.iget(0).iget(0)._group
    assert g2._lvl == 3


def test_group_info(ecoli_core):
    m1 = ecoli_core
    g2 = Group({"ecoli_g1": {"g11": {"m111": m1, "m112": m1}, "m12": m1},
                "ecoli_g2": {"m21": m1, "m22": m1},
                "m3": m1}, name_tag="G2")
    assert g2.size == 6
    assert len(g2.reaction_ids) == len(m1.reactions)
    assert len(g2.metabolite_ids) == len(m1.metabolites)
    assert len(g2.gene_ids) == len(m1.genes)
    assert isinstance(g2.get_info(), pd.DataFrame), g2.get_info()


def test_group_get_info_2(ecoli_core):
    m1 = ecoli_core
    g2 = Group({"ecoli_g1": {"g11": {"m111": m1, "m112": m1}, "m12": m1},
                "ecoli_g2": {"m21": m1, "m22": m1},
                "m3": m1}, name_tag="G2")
    print(g2.get_info(features=["n_rxns", "n_mets", "n_genes"]))


def test_group_get_flux(ecoli_core):
    m1 = ecoli_core
    g2 = Group({"ecoli_g1": {"e11": m1, "e12": m1}, "ecoli_g2": {"e21": m1, "e22": m1}, "a": m1}, name_tag="G2")
    fba_result = g2.do_flux_analysis(method="FBA", solver="glpk")
    sampling_result = g2.do_flux_analysis(method="sampling", solver="glpk", n=10)
    assert isinstance(fba_result.result, pd.DataFrame)
    assert isinstance(sampling_result.result, dict)


def test_group_operations(ecoli_core):
    m1 = ecoli_core
    g = Group({"ecoli_g1": {"e11": m1, "e12": m1, "e13": m1}, "ecoli_g2": {"e21": m1, "e22": m1}, "a": m1}, name_tag="G2")
    assert len(g["ecoli_g1"])==3
    assert isinstance(g["ecoli_g1"]["e11"], Model)
    g["ecoli_g3"] = {"e31": m1, "e32": m1}
    assert len(g["ecoli_g3"])==2
    for gi in g:
        print(gi.name_tag)


def test_compare_sim(ecoli_core):
    m1 = ecoli_core
    g = Group(group={"ecoli_g1": {"e11": m1, "e12": m1, "e13": m1},
                     "ecoli_g2": {"e21": m1, "e22": m1}, "a": m1},
              name_tag="G2")
    sim_comp = g.compare(tags=None, compare_models=True, use="jaccard")
    sim_comp.plot(dpi=150)


def test_compare_num(ecoli_core):
    m1 = ecoli_core
    g = Group(group={"ecoli_g1": {"e11": m1, "e12": m1, "e13": m1},
                     "ecoli_g2": {"e21": m1, "e22": m1}, "a": m1},
              name_tag="G2")
    num_comp = g.compare(tags=None, compare_models=True, use="num")
    print(num_comp.result)
    print(num_comp.name_order)
    num_comp.plot(dpi=150)

    num_comp = g.compare(tags=None, compare_models=False, use="num")
    print(num_comp.result)
    print(num_comp.name_order)
    num_comp.plot(dpi=150, group="model", name_order=[gi.name_tag for gi in g])


# TODO: test_add_tasks
