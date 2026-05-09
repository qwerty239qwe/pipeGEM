import argparse
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pipeGEM.cli.cli import (
    main, run_pipeline, _build_parser, _is_legacy_invocation,
    _parse_legacy, _run_legacy,
)
from pipeGEM.cli._validation import validate_pipeline_configs


# ---------------------------------------------------------------------------
# --version
# ---------------------------------------------------------------------------

def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "pipeGEM" in captured.out


# ---------------------------------------------------------------------------
# No args → help
# ---------------------------------------------------------------------------

def test_no_args_shows_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "subcommand" in captured.out.lower() or "pipeGEM" in captured.out


# ---------------------------------------------------------------------------
# Subcommand help texts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("subcmd", ["template", "integrate", "process",
                                     "threshold", "flux", "compare"])
def test_subcommand_help(subcmd, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main([subcmd, "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert len(captured.out) > 0


# ---------------------------------------------------------------------------
# template subcommand
# ---------------------------------------------------------------------------

def test_template_subcommand_creates_configs(tmp_path):
    output_dir = str(tmp_path / "out")
    with patch("pipeGEM.cli.cli.validate_pipeline_configs"):
        with patch("pipeGEM.cli._utils.generate_template_configs") as mock_gen:
            main(["template", "-p", "integration", "-o", output_dir])
            mock_gen.assert_called_once_with(dest_folder=output_dir,
                                             pl_name="integration")


def test_template_missing_required_flag(capsys):
    """template requires both -p and -o; omitting them should error."""
    with pytest.raises(SystemExit) as exc_info:
        main(["template"])
    assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# integrate subcommand — missing configs
# ---------------------------------------------------------------------------

def test_integrate_missing_configs_errors(capsys):
    """integrate without required flags raises a clear error."""
    with pytest.raises(SystemExit) as exc_info:
        main(["integrate"])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error" in captured.err
    assert "integrate" in captured.err


# ---------------------------------------------------------------------------
# process subcommand — missing config
# ---------------------------------------------------------------------------

def test_process_missing_config_errors(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["process"])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error" in captured.err
    assert "process" in captured.err


# ---------------------------------------------------------------------------
# compare subcommand — missing config
# ---------------------------------------------------------------------------

def test_compare_missing_config_errors(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["compare"])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error" in captured.err
    assert "compare" in captured.err


# ---------------------------------------------------------------------------
# Validation — file does not exist
# ---------------------------------------------------------------------------

def test_validate_pipeline_configs_missing_file(capsys):
    args = argparse.Namespace(model_testing_conf_path="/nonexistent/model.toml")
    with pytest.raises(SystemExit) as exc_info:
        validate_pipeline_configs("process", args)
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_validate_pipeline_configs_all_present(tmp_path):
    """When all required config files exist, validation passes silently."""
    model_file = tmp_path / "model.toml"
    model_file.write_text("[input]\npath = 'test'\n")
    args = argparse.Namespace(model_testing_conf_path=str(model_file))
    # Should not raise
    validate_pipeline_configs("process", args)


# ---------------------------------------------------------------------------
# Legacy invocation detection
# ---------------------------------------------------------------------------

def test_is_legacy_invocation():
    assert _is_legacy_invocation(["-n", "integration", "-g", "gene.toml"])
    assert _is_legacy_invocation(["--name", "integration"])
    assert not _is_legacy_invocation(["integrate", "-g", "gene.toml"])
    assert not _is_legacy_invocation(["--version"])


def test_legacy_invocation_warns():
    with patch("pipeGEM.cli.cli.run_pipeline") as mock_run:
        with patch("pipeGEM.cli._utils.read_configs", return_value={}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                main(["-n", "template", "-p", "integration", "-o", "/tmp/out"])
                assert len(w) == 1
                assert issubclass(w[0].category, FutureWarning)
                assert "deprecated" in str(w[0].message).lower()


def test_legacy_invocation_still_works():
    with patch("pipeGEM.cli.cli.run_pipeline") as mock_run:
        with patch("pipeGEM.cli._utils.read_configs", return_value={}):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                main(["-n", "template", "-p", "integration", "-o", "/tmp/out"])
                mock_run.assert_called_once()
                call_kwargs = mock_run.call_args
                assert call_kwargs[1].get("pipeline") == "integration" or \
                       call_kwargs.kwargs.get("pipeline") == "integration"


# ---------------------------------------------------------------------------
# run_pipeline elif — only one branch executes
# ---------------------------------------------------------------------------

def test_run_pipeline_elif_only_one_branch():
    """Verify that run_pipeline uses elif and only one branch executes."""
    with patch("pipeGEM.cli._utils.generate_template_configs") as mock_template, \
         patch("pipeGEM.cli._utils.run_integration_pipeline") as mock_integrate:
        run_pipeline("template", output_path="/tmp", pipeline="integration")
        mock_template.assert_called_once()
        mock_integrate.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling in main()
# ---------------------------------------------------------------------------

def test_file_not_found_error_handling(tmp_path, capsys):
    """FileNotFoundError is caught and printed as user-friendly message."""
    with patch("pipeGEM.cli.cli.validate_pipeline_configs"):
        with patch("pipeGEM.cli._utils.read_configs",
                   side_effect=FileNotFoundError("missing.toml")):
            with pytest.raises(SystemExit) as exc_info:
                main(["process", "-t", str(tmp_path / "fake.toml")])
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "file not found" in captured.err.lower()


def test_key_error_handling(tmp_path, capsys):
    """KeyError from bad TOML is caught and printed clearly."""
    fake_conf = tmp_path / "bad.toml"
    fake_conf.write_text("[section]\nkey = 'value'\n")
    with patch("pipeGEM.cli.cli.validate_pipeline_configs"):
        with patch("pipeGEM.cli._utils.read_configs",
                   side_effect=KeyError("missing_key")):
            with pytest.raises(SystemExit) as exc_info:
                main(["process", "-t", str(fake_conf)])
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "missing config key" in captured.err.lower()
