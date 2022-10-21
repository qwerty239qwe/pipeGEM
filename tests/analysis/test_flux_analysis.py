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
                "ecoli_g2": {"e21": random_perturb(m1.copy()), "e22": m1}}, name_tag="G2")
    pFBA_result = g2.do_flux_analysis(method="FBA", solver="glpk")
    print(pFBA_result.result)
    print(pFBA_result)
    yield pFBA_result


def test_dim_reduction(pFBA_result):
    pca = pFBA_result.dim_reduction()
    print(pca.result)
    pca.plot(dpi=150)

    tsne = pFBA_result.dim_reduction(method="TSNE")
    tsne.plot(dpi=150)


def test_corr(pFBA_result):
    corr_result = pFBA_result.corr("name")
    corr_result.plot()

    corr_result = pFBA_result.corr("reaction")
    corr_result.plot()