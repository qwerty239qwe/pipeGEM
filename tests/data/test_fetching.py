from pipeGEM.data import fetching


def test_list_models():
    result = fetching.list_models()

    assert "metabolic atlas" in result["database"]
    assert "BiGG" in result["database"]