import pytest
import cobra
from pipeGEM.data.synthesis import get_syn_gene_data
from pipeGEM.utils import load_model
from pipeGEM.data.fetching import load_remote_model
from pipeGEM.utils import random_perturb
from pipeGEM import Group


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