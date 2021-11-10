import pytest
import cobra


@pytest.fixture(scope="session")
def ecoli_core():
    return cobra.io.load_model(model_id="e_coli_core")


@pytest.fixture(scope="session")
def yeast():
    return cobra.io.load_model(model_id="iND750")


@pytest.fixture(scope="session")
def ecoli():
    return cobra.io.load_model(model_id="iML1515")