from pathlib import Path
from unittest.mock import patch

import pytest

from pipeGEM.cli.cli import main
from pipeGEM.cli.config import (
    FluxPipelineConfig,
    IntegrationPipelineConfig,
    ModelProcessingPipelineConfig,
    ThresholdPipelineConfig,
)
from pipeGEM.cli.config.base import ConfigSource
from pipeGEM.cli.config.types import FluxAnalysisConfig, GeneDataConfig, MultiModelConfig
from pipeGEM.cli.errors import ConfigError, PipelineError
from pipeGEM.cli.pipelines import FluxPipeline, ModelProcessingPipeline


CONFIG_ROOT = Path(__file__).parents[2] / "pipeGEM" / "cli" / "configs"


def test_bundled_configs_load_into_pipeline_config_objects():
    config = IntegrationPipelineConfig.from_files(
        gene_data=CONFIG_ROOT / "gene_data.toml",
        model=CONFIG_ROOT / "model.toml",
        threshold=CONFIG_ROOT / "thresholds" / "percentile_threshold.toml",
        mapping=CONFIG_ROOT / "mapping.toml",
        integration=CONFIG_ROOT / "gene_data_integration" / "rFASTCORMICS.toml",
    )

    assert config.gene_data.input_file_path == Path("")
    assert config.threshold.name == "percentile"
    assert config.integration.integrator_name == "rFASTCORMICS"


def test_bundled_integration_configs_load_with_nested_precompute_keys():
    for path in (CONFIG_ROOT / "gene_data_integration").glob("*.toml"):
        config = IntegrationPipelineConfig.from_files(
            gene_data=CONFIG_ROOT / "gene_data.toml",
            model=CONFIG_ROOT / "model.toml",
            threshold=CONFIG_ROOT / "thresholds" / "percentile_threshold.toml",
            mapping=CONFIG_ROOT / "mapping.toml",
            integration=path,
        )
        assert config.integration.require("precompute.model.cobra_model_path") is None
        assert config.integration.require("precompute.tasks.task_result_path") is None
        assert config.integration.require("precompute.threshold.threshold_result_path") is None


def test_config_as_dict_is_defensive_copy():
    config = GeneDataConfig.from_toml(CONFIG_ROOT / "gene_data.toml", "gene_data")

    first = config.as_dict()
    first["input"].pop("input_file_path")

    second = config.as_dict()
    assert "input_file_path" in second["input"]


def test_missing_required_config_key_reports_path(tmp_path):
    bad = tmp_path / "bad_gene.toml"
    bad.write_text("[input]\nsep = '\\t'\n[params]\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        GeneDataConfig.from_toml(bad, "gene_data")

    assert "missing config key 'input.input_file_path'" in str(exc_info.value)
    assert str(bad) in str(exc_info.value)


def test_pipeline_run_passes_defensive_copy_to_legacy_helper():
    config = ModelProcessingPipelineConfig.from_files(CONFIG_ROOT / "model.toml")
    pipeline = ModelProcessingPipeline(config)

    with patch("pipeGEM.cli._utils.preprocess_model") as mock_preprocess:
        pipeline.run()

    called_conf = mock_preprocess.call_args.kwargs["model_conf"]
    called_conf["input"].pop("input_file_path")
    assert "input_file_path" in config.model.as_dict()["input"]


def test_process_dry_run_does_not_execute_pipeline(capsys):
    with patch("pipeGEM.cli._utils.preprocess_model") as mock_preprocess:
        main(["process", "-t", str(CONFIG_ROOT / "model.toml"), "--dry-run"])

    captured = capsys.readouterr()
    assert "Pipeline: model_processing" in captured.out
    mock_preprocess.assert_not_called()


def test_threshold_plan_uses_gene_input_and_threshold_output():
    config = ThresholdPipelineConfig.from_files(
        gene_data=CONFIG_ROOT / "gene_data.toml",
        threshold=CONFIG_ROOT / "thresholds" / "percentile_threshold.toml",
    )
    from pipeGEM.cli.pipelines import ThresholdPipeline

    plan = ThresholdPipeline(config).plan()
    assert plan.steps[0].inputs == (Path(""),)
    assert plan.steps[0].outputs == (Path(""),)


def test_flux_partial_integration_configs_error(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main([
            "flux",
            "-f", str(CONFIG_ROOT / "flux_analysis" / "pFBA.toml"),
            "-t", str(CONFIG_ROOT / "multi_model.toml"),
            "-i", str(CONFIG_ROOT / "gene_data_integration" / "rFASTCORMICS.toml"),
            "--dry-run",
        ])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "requires all integration configs" in captured.err


def test_flux_pipeline_run_validates_model_config_before_legacy_call():
    config = FluxPipelineConfig(
        flux_analysis=FluxAnalysisConfig(
            raw={"saved_path": "flux_result"},
            source=ConfigSource("flux_analysis"),
        ),
        model=MultiModelConfig(
            raw={"model_type": "pg", "models_input_dir": "models"},
            source=ConfigSource("model"),
        ),
    )
    pipeline = FluxPipeline(config)

    with patch("pipeGEM.cli._utils.do_flux_analysis") as mock_flux:
        with pytest.raises(ConfigError) as exc_info:
            pipeline.run()

    assert "group_factor_path" in str(exc_info.value)
    mock_flux.assert_not_called()


def test_required_key_presence_allows_toml_none_values():
    config = GeneDataConfig(
        raw={"input": {"input_file_path": None}, "params": {}},
        source=ConfigSource("gene_data"),
    )

    assert config.require("input.input_file_path") is None


def test_template_dry_run_rejects_unknown_pipeline(capsys, tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        main(["template", "-p", "unknown", "-o", str(tmp_path), "--dry-run"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "unknown template pipeline" in captured.err
