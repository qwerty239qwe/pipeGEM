import sys
from pathlib import Path


# Maps each pipeline to its required config argument names and flag strings.
_PIPELINE_REQUIRED_CONFIGS = {
    "integrate": {
        "gene_data_conf_path": "--gene_data (-g)",
        "model_testing_conf_path": "--model_testing (-t)",
        "threshold_conf_path": "--threshold (-r)",
        "mapping_conf_path": "--mapping (-m)",
        "integration_conf_path": "--integration (-i)",
    },
    "process": {
        "model_testing_conf_path": "--model_testing (-t)",
    },
    "threshold": {
        "gene_data_conf_path": "--gene_data (-g)",
        "threshold_conf_path": "--threshold (-r)",
    },
    "flux": {
        "flux_analysis_conf_path": "--flux_analysis (-f)",
        "model_testing_conf_path": "--model_testing (-t)",
    },
    "compare": {
        "comparison_conf_path": "--comparison (-c)",
    },
    "template": {
        "pipeline": "--pipeline (-p)",
        "output_path": "--output (-o)",
    },
}


def validate_pipeline_configs(subcommand, args):
    """Validate that required config files are provided and exist on disk.

    Parameters
    ----------
    subcommand : str
        The subcommand name (e.g., "integrate", "process").
    args : argparse.Namespace
        Parsed arguments from argparse.

    Raises
    ------
    SystemExit
        If a required config is missing or a file does not exist.
    """
    required = _PIPELINE_REQUIRED_CONFIGS.get(subcommand, {})

    for attr, flag in required.items():
        value = getattr(args, attr, None)
        if value is None:
            print(f"Error: pipeline '{subcommand}' requires {flag} config file.",
                  file=sys.stderr)
            sys.exit(1)

        # template args (-p, -o) are not file paths to validate on disk
        if subcommand == "template":
            continue

        path = Path(value)
        if not path.is_file():
            print(f"Error: config file not found: {value}", file=sys.stderr)
            sys.exit(1)
