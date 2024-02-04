import numpy as np
import pandas as pd
import pytest
import cobra
from pipeGEM.data.synthesis import get_syn_gene_data
from pipeGEM.data.fetching import load_remote_model
from pipeGEM import Group
from pipeGEM.utils import random_perturb, load_model
from pipeGEM.analysis.tasks import Task, TaskContainer
from pipeGEM.analysis import FBA_Analysis
from pandas.api.types import is_numeric_dtype


@pytest.fixture(scope="session")
def ecoli_core():
    return cobra.io.load_model(model_id="e_coli_core")


@pytest.fixture(scope="session")
def yeast():
    return cobra.io.load_model(model_id="iND750")


@pytest.fixture(scope="session")
def ecoli():
    return cobra.io.load_model(model_id="iML1515")


@pytest.fixture(scope="session")
def ecoli_core_data(ecoli_core):
    return get_syn_gene_data(ecoli_core, n_sample=100)


@pytest.fixture(scope="session")
def Human_GEM():
    return load_remote_model(model_id="Human-GEM")


@pytest.fixture(scope="session")
def Human_GEM_data(Human_GEM):
    return get_syn_gene_data(Human_GEM, n_sample=10)


@pytest.fixture(scope="session")
def group(ecoli_core):
    return Group({"ecoli_g1": {"m111": random_perturb(ecoli_core, on_structure=False, constr_ratio=0.8, random_state=0),
                               "m112": random_perturb(ecoli_core, on_structure=False, constr_ratio=0.8, random_state=1),
                               "m12": random_perturb(ecoli_core, on_structure=False, constr_ratio=0.8, random_state=2)},
                    "ecoli_g2": {"m21": random_perturb(ecoli_core, on_structure=False, constr_ratio=0.7, random_state=3),
                                 "m22": random_perturb(ecoli_core, on_structure=False, constr_ratio=0.7, random_state=4)},
                    "ecoli_g3": {"m3": ecoli_core}}, name_tag="G2",
                    treatment={"m111": "a", "m112": "b", "m12": "b"})


@pytest.fixture(scope="session")
def pFBA_result(ecoli_core) -> FBA_Analysis:
    m1 = ecoli_core
    g2 = Group({"ecoli_g1": {"e11": m1, "e12": random_perturb(m1.copy())},
                "ecoli_g2": {"e21": random_perturb(m1.copy()), "e22": m1}},
               name_tag="G2",
               treatments={"e11": "A", "e12": "B", "e21": "B", "e22": "A"})
    pFBA_result = g2.do_flux_analysis(method="FBA",
                                      solver="glpk",)
    flux_df = pFBA_result.flux_df
    num_cols = [c for c in flux_df.columns if is_numeric_dtype(c)]
    noise = pd.DataFrame(data=np.random.random(size=(flux_df.shape[0], len(num_cols))),
                         index=flux_df.index,
                         columns=num_cols)
    pFBA_result._result["flux_df"].loc[:, num_cols] += noise
    yield pFBA_result


@pytest.fixture(scope="session")
def ecoli_Tasks():
    new_task = Task(should_fail=False,
                    in_mets=[{"met_id": "glc__D", "compartment": "e", "lb": 0, "ub": 10},
                             {"met_id": "o2", "compartment": "e", "lb": 0, "ub": 10},
                             {"met_id": "h2o", "compartment": "e", "lb": 0, "ub": 10},
                             {"met_id": "h", "compartment": "e", "lb": 0, "ub": 10}],
                    out_mets=[{"met_id": "co2", "compartment": "e", "lb": 0, "ub": 10},
                              {"met_id": "h2o", "compartment": "e", "lb": 0, "ub": 10},
                              {"met_id": "atp", "compartment": "c", "lb": 1, "ub": 10},
                              {"met_id": "h", "compartment": "e", "lb": 0, "ub": 10}],
                    compartment_parenthesis="_{}"
                    )
    yield TaskContainer({"glu2atp": new_task})