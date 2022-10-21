import pandas as pd
import cobra

from pipeGEM.core import Model
from pipeGEM.core import Group

from pipeGEM.utils import random_perturb


def test_flux_analysis(ecoli_core):
    m1 = ecoli_core
    g2 = Group({"ecoli_g1": {"e11": m1, "e12": random_perturb(m1.copy())},
                "ecoli_g2": {"e21": random_perturb(m1.copy()), "e22": m1}}, name_tag="G2")
    pFBA_result = g2.do_flux_analysis(method="FBA", solver="glpk")
    print(pFBA_result.result)
    print(pFBA_result)
    pca = pFBA_result.dim_reduction()
    print(pca.result)
    pca.plot(dpi=150)