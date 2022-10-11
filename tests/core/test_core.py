import pandas as pd
import cobra

from pipeGEM.core import Model
from pipeGEM.core import Group


def test_get_model(ecoli_core):
    return Model.from_cobra_model(model=ecoli_core, name="ecoli")


def test_get_group(ecoli_core):
    m1 = ecoli_core
    g = Group({"ecoli": m1, "ecoli_2": m1.copy()}, name_tag="G1")
    assert m1 == g.tget("ecoli")[0].cobra_model
    assert isinstance(g.tget("ecoli")[0].model, Model)
    assert g._lvl == 2

    g2 = Group({"ecoli_g1": {"e11": m1, "e12": m1}, "ecoli_g2": {"e21": m1, "e22": m1}, "a": m1}, name_tag="G2")
    assert g2.iget([0, 0])[0].name_tag == "e11"
    assert g2._lvl == 3


def test_group_info(ecoli_core):
    m1 = ecoli_core
    g2 = Group({"ecoli_g1": {"e11": m1, "e12": m1}, "ecoli_g2": {"e21": m1, "e22": m1}, "a": m1}, name_tag="G2")
    assert isinstance(g2.get_info(), pd.DataFrame), g2.get_info()
    assert len(g2.reaction_ids) == len(m1.reactions)
    assert len(g2.metabolite_ids) == len(m1.metabolites)
    assert len(g2.gene_ids) == len(m1.genes)
    assert all([isinstance(g, Model) for g in g2.get_info(features=["model"])["model"]]), \
        g2.get_info(features=["model"])


def test_group_get_flux(ecoli_core):
    m1 = ecoli_core
    g2 = Group({"ecoli_g1": {"e11": m1, "e12": m1}, "ecoli_g2": {"e21": m1, "e22": m1}, "a": m1}, name_tag="G2")
    fba_result = g2.do_flux_analysis(method="FBA")
    sampling_result = g2.do_flux_analysis(method="sampling", n=10)


def test_group_operations(ecoli_core):
    m1 = ecoli_core

    g = Group({"ecoli_g1": {"e11": m1, "e12": m1, "e13": m1}, "ecoli_g2": {"e21": m1, "e22": m1}, "a": m1}, name_tag="G2")
    assert len(g["ecoli_g1"])==3
    assert isinstance(g["ecoli_g1"]["e11"], Model)
    g["ecoli_g3"] = {"e31": m1, "e32": m1}
    assert len(g["ecoli_g3"])==2
    for gi in g:
        print(gi.name_tag)
