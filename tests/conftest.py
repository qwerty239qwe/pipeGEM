import pytest
import cobra


@pytest.fixture(scope="session")
def ecoli_core():
    return cobra.io.load_model(model_id="e_coli_core")