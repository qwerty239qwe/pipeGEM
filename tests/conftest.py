import pytest
import cobra
from pipeGEM.data.synthesis import get_syn_gene_data


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