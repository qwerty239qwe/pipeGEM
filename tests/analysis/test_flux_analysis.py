import pandas as pd
import cobra
import pytest

from pipeGEM.core import Model
from pipeGEM.core import Group

from pipeGEM.utils import random_perturb


@pytest.fixture(scope="session")
def pFBA_result(ecoli_core):
    m1 = ecoli_core
    g2 = Group({"ecoli_g1": {"e11": m1, "e12": random_perturb(m1.copy())},
                "ecoli_g2": {"e21": random_perturb(m1.copy()), "e22": m1}}, name_tag="G2",
                treatments={"e11": "A", "e12": "B", "e21": "B", "e22": "A"})
    pFBA_result = g2.do_flux_analysis(method="FBA", solver="glpk", group_by="treatments")
    yield pFBA_result


def test_PCA(pFBA_result):
    pca = pFBA_result.dim_reduction()
    pca.plot(dpi=150, color_by="treatments")


def test_TSNE(pFBA_result):
    tsne = pFBA_result.dim_reduction(method="TSNE", n_components=2)
    tsne.plot(dpi=150, color_by="treatments")


def test_UMAP(pFBA_result):
    tsne = pFBA_result.dim_reduction(method="UMAP", n_components=2)
    tsne.plot(dpi=150, color_by="treatments")


def test_corr(pFBA_result):
    corr_result = pFBA_result.corr("treatments")
    corr_result.plot()

    corr_result = pFBA_result.corr("reaction")
    corr_result.plot()