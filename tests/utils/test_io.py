from unittest.mock import Mock

import cobra
import pytest

from pipeGEM import Model
from pipeGEM.utils import load_model, save_toml_file


@pytest.mark.parametrize("suffix, reader_name", [
    (".xml", "read_sbml_model"),
    (".json", "load_json_model"),
    (".mat", "load_matlab_model"),
    (".yaml", "load_yaml_model"),
    (".yml", "load_yaml_model"),
])
@pytest.mark.parametrize("case", ["lower", "upper", "mixed"])
def test_load_model_dispatches_case_insensitively_without_changing_filename(
        tmp_path, monkeypatch, suffix, reader_name, case):
    extension = suffix if case == "lower" else suffix.upper() if case == "upper" else suffix.title()
    path = tmp_path / f"Recon{extension}"
    result = object()
    reader = Mock(return_value=result)
    monkeypatch.setattr(cobra.io, reader_name, reader)

    assert load_model(path) is result
    reader.assert_called_once_with(path if suffix in (".yaml", ".yml") else str(path))


@pytest.mark.parametrize("suffix", [".json", ".JSON", ".JsOn"])
@pytest.mark.parametrize("wrapped", [False, True], ids=["cobra", "pipegem"])
def test_json_model_loads_real_data_and_preserves_sidecar_metadata(
        tmp_path, trivial_linear_model, suffix, wrapped):
    path = tmp_path / f"Recon{suffix}"
    cobra.io.save_json_model(trivial_linear_model, str(path))
    save_toml_file(tmp_path / "Recon_annotations.toml",
                   {"name_tag": "annotated_sample", "condition": "control"})

    loaded = Model.load_model(path) if wrapped else load_model(str(path))
    assert loaded.id == "trivial_linear"
    assert loaded.reactions.BIOMASS.objective_coefficient == 1
    # The fixture's uptake limit bounds biomass production at 10.
    assert loaded.slim_optimize() == pytest.approx(10.)
    if wrapped:
        assert loaded.name_tag == "annotated_sample"
        assert loaded.annotation["condition"] == "control"


@pytest.mark.parametrize("suffix", [".txt", ".CSV", ""])
def test_load_model_still_rejects_unsupported_extensions(tmp_path, suffix):
    with pytest.raises(ValueError, match="Invalid file extension") as error:
        load_model(tmp_path / f"Recon{suffix}")
    assert str(error.value) == f"Invalid file extension: {suffix}"
