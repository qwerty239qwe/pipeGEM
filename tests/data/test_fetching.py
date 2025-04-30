from pipeGEM.data import fetching
import pytest
import os


def test_list_models():
    result = fetching.list_models()
    # assert "metabolic atlas" in result["database"].to_list()  # "metabolic atlas" cannot be fetch in github actions
    assert "BiGG" in result["database"].to_list()


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true",
                   reason="Skip in CI environment due to API restrictions")
def test_metabolic_atlas_exists():
    """Test that metabolic atlas API is accessible and returns data."""
    result = fetching.list_models()
    metabolic_atlas_models = result[result["database"] == "metabolic atlas"]
    assert not metabolic_atlas_models.empty
    # Additional checks to verify the data structure
    assert "model_id" in metabolic_atlas_models.columns
    assert "description" in metabolic_atlas_models.columns
    assert len(metabolic_atlas_models) > 0  # Ensure at least one model exists