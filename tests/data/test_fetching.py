from pipeGEM.data import fetching


def test_list_models():
    result = fetching.list_models()

    assert "metabolic atlas" in result["database"].to_list()
    assert "BiGG" in result["database"].to_list()