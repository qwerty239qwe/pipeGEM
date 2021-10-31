import pandas as pd

from pipeGEM.core._model import Model
from pipeGEM.core._group import Group


def test_get_model(ecoli_core):
    return Model(model=ecoli_core, name_tag="ecoli")


def test_get_group(ecoli_core):
    m1 = ecoli_core
    g = Group({"ecoli": m1, "ecoli_2": m1.copy()}, name_tag="G1")
    assert m1 == g.tget("ecoli")[0].model
    assert g._lvl == 1

    g2 = Group({"ecoli_g1": {"e11": m1, "e12": m1}, "ecoli_g2": {"e21": m1, "e22": m1}, "a": m1}, name_tag="G2")
    assert g2.iget([0, 0])[0].name_tag == "e11"
    assert g2._lvl == 2

    assert isinstance(g2.get_info(), pd.DataFrame), g2.get_info()
    assert len(g2.reaction_ids) == len(m1.reactions)
    assert len(g2.metabolite_ids) == len(m1.metabolites)
    assert len(g2.gene_ids) == len(m1.genes)

    assert all([isinstance(g, Model) for g in g2.get_models()])
