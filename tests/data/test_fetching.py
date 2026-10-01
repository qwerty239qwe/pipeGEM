from pipeGEM.data import fetching
import pytest
import os


def test_list_models():
    result = fetching.list_models()
    if result.empty:
        pytest.skip("Live model registry APIs are unavailable in this environment")
    # assert "metabolic atlas" in result["database"].to_list()  # "metabolic atlas" cannot be fetch in github actions
    assert "BiGG" in result["database"].to_list()


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true",
                   reason="Skip in CI environment due to API restrictions")
def test_metabolic_atlas_exists():
    """Test that metabolic atlas API is accessible and returns data."""
    result = fetching.list_models()
    if result.empty:
        pytest.skip("Live model registry APIs are unavailable in this environment")
    metabolic_atlas_models = result[result["database"] == "metabolic atlas"]
    assert not metabolic_atlas_models.empty
    # schema produced by AtlasDataBaseFetcher.manipulate_df
    for col in ["id", "organism", "reaction_count", "metabolite_count", "gene_count"]:
        assert col in metabolic_atlas_models.columns
