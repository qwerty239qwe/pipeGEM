from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from pipeGEM import Model, Group
from pipeGEM.analysis import FBA_Analysis, FVA_Analysis, TaskAnalysis
from pipeGEM.cli._utils import do_flux_analysis, do_model_comparison
from pipeGEM.utils import save_model


@pytest.mark.parametrize("model_type", ["pg", "cobra"])
@pytest.mark.parametrize("suffix", [".json", ".JSON", ".JsOn"])
def test_flux_loads_models_and_named_factors(tmp_path, trivial_linear_model, model_type, suffix):
    models = tmp_path / "models"
    models.mkdir()
    if model_type == "pg":
        Model(name_tag="sample", model=trivial_linear_model).save_model(models / "sample.json")
    else:
        save_model(trivial_linear_model, models / "sample.json")
    (models / "sample.json").rename(models / f"sample{suffix}")
    (models / "notes.txt").write_text("not a model", encoding="utf-8")
    (models / "subdir").mkdir()
    factors = tmp_path / "factors.csv"
    pd.DataFrame({"condition": ["control"]}, index=["sample"]).to_csv(factors)
    fa = {"method": "FBA", "solver": "glpk", "saved_path": str(tmp_path / "flux")}
    do_flux_analysis(fa.copy(), {
        "models_input_dir": str(models), "model_type": model_type,
        "group_factor_path": str(factors),
    }, None, None, None, None)
    loaded = FBA_Analysis.load(tmp_path / "flux")
    assert set(loaded.flux_df["model"]) == {"sample"}
    assert set(loaded.flux_df["condition"]) == {"control"}
    biomass = loaded.flux_df.query("Reaction == 'BIOMASS'")
    assert biomass["fluxes"].iloc[0] == pytest.approx(10.)


@pytest.mark.parametrize("model_type", ["pg", "cobra"])
@pytest.mark.parametrize("suffix", [".json", ".JSON", ".JsOn"])
def test_comparison_ignores_sidecars_and_directories(
        tmp_path, trivial_linear_model, monkeypatch, model_type, suffix):
    models = tmp_path / "models"
    models.mkdir()
    Model(name_tag="sample", model=trivial_linear_model).save_model(models / "sample.json")
    (models / "sample.json").rename(models / f"sample{suffix}")
    (models / "notes.txt").write_text("not a model", encoding="utf-8")
    (models / "subdir").mkdir()
    seen = []

    def compare(group, **kwargs):
        seen.append(list(group._group))
        return SimpleNamespace(plot=lambda **kwargs: None)

    monkeypatch.setattr(Group, "compare", compare)
    do_model_comparison({
        "models_input_path": str(models), "model_type": model_type,
        "factor_file": str(tmp_path / "absent.csv"), "output_dir": str(tmp_path / "comparison"),
        "compare_num": {"group": None, "output_file_name": "num.png", "dpi": 72},
        "compare_jaccard": {"row_color_by": None, "col_color_by": None, "output_file_name": "jaccard.png", "dpi": 72},
        "compare_PCA": {"color_by": None, "output_file_name": "pca.png", "dpi": 72},
    })
    assert seen == [["sample"]] * 3


def test_flux_with_integration_runs_fva_and_reuses_saved_models(tmp_path, trivial_linear_model):
    model_path = tmp_path / "sample.json"
    save_model(trivial_linear_model, model_path)
    genes = tmp_path / "genes.csv"
    pd.DataFrame({"sample": [1., 4., 4., 2.]}, index=["g1", "g2", "g3", "g4"]).to_csv(genes)
    tasks = TaskAnalysis(log={})
    tasks.add_result({"result_df": pd.DataFrame({"Passed": [False], "task_support_rxns": [[]]}, index=["unused"])})
    tasks.save(tmp_path / "tasks")
    threshold_dir = tmp_path / "thresholds"
    gene_conf = {"input": {"input_file_path": str(genes), "index_col": 0}, "params": {}}
    threshold_conf = {"saved_path": str(threshold_dir), "params": {"name": "percentile", "p": [25, 75]}}
    mapping_conf = {
        "threshold_analysis": {"type": "percentile", "input_file_path_pattern": str(threshold_dir / "{data_name}")},
        "rxn_score": {"align": {}},
        "task_score": {"input_file_path": str(tmp_path / "tasks"), "get_supp_rxns": {}},
    }
    integration_conf = {
        "integrator_name": "EFlux", "saved_path": str(tmp_path / "integration" / "{}"),
        "max_ub": 10., "min_lb": 1., "transform": "exp",
        "precompute": {"threshold": {"threshold_result_path": None, "type": "percentile"}},
    }
    model_conf = {"models_input_path": str(tmp_path / "{}.json"), "model_type": "cobra"}
    for run in ("cold", "cached"):
        do_flux_analysis(
            {"method": "FVA", "solver": "glpk", "is_loopless": False, "saved_path": str(tmp_path / run)},
            model_conf.copy(), deepcopy(gene_conf), deepcopy(threshold_conf),
            deepcopy(mapping_conf), deepcopy(integration_conf),
        )
        loaded = FVA_Analysis.load(tmp_path / run)
        assert set(loaded.flux_df["model"]) == {"sample"}
        assert loaded.flux_df.query("Reaction == 'R1'")["maximum"].iloc[0] == pytest.approx(1.)
    assert (tmp_path / "integration/sample/result/result_model.json").is_file()
