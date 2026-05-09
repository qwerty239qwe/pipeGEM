import argparse
import sys
import warnings
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

from pipeGEM.cli.config import (
    ComparisonPipelineConfig,
    FluxPipelineConfig,
    IntegrationPipelineConfig,
    ModelProcessingPipelineConfig,
    TemplatePipelineConfig,
    ThresholdPipelineConfig,
)
from pipeGEM.cli._doc import PROGRAM_DESCRIPTION, _docs
from pipeGEM.cli._validation import validate_pipeline_configs
from pipeGEM.cli.errors import CLIError
from pipeGEM.cli.pipelines import (
    ComparisonPipeline,
    FluxPipeline,
    IntegrationPipeline,
    ModelProcessingPipeline,
    TemplatePipeline,
    ThresholdPipeline,
)


# ---------------------------------------------------------------------------
# Legacy run_pipeline (kept for backward-compatible -n invocations)
# ---------------------------------------------------------------------------

def run_pipeline(pl_name, **configs):
    """Main pipeline execution function (legacy path)."""
    from pipeGEM.cli._utils import (
        preprocess_model, find_threshold, read_configs,
        generate_template_configs, run_integration_pipeline,
        do_model_comparison, do_flux_analysis,
    )
    from pipeGEM.cli._io import load_gene_data

    if pl_name == "template":
        generate_template_configs(dest_folder=configs.get("output_path"),
                                  pl_name=configs.get("pipeline"))
    elif pl_name == "integration":
        run_integration_pipeline(gene_data_conf=configs.get("gene_data_conf"),
                                 model_conf=configs.get("model_conf"),
                                 threshold_conf=configs.get("threshold_conf"),
                                 mapping_conf=configs.get("mapping_conf"),
                                 integration_conf=configs.get("integration_conf"))
    elif pl_name == "model_processing":
        _, _ = preprocess_model(model_conf=configs.get("model_conf"))
    elif pl_name == "get_threshold":
        gene_data_dic = load_gene_data(gene_data_conf=configs.get("gene_data_conf"))
        _ = find_threshold(gene_data_dic, configs.get("threshold_conf"))
    elif pl_name == "do_flux_analysis":
        do_flux_analysis(fa_configs=configs.get("fa_conf"),
                         multi_model_conf=configs.get("model_conf"),
                         gene_data_conf=configs.get("gene_data_conf"),
                         threshold_conf=configs.get("threshold_conf"),
                         mapping_conf=configs.get("mapping_conf"),
                         integration_conf=configs.get("integration_conf"))
    elif pl_name == "do_model_comparison":
        do_model_comparison(comparison_configs=configs.get("comparison_conf"))
    elif pl_name == "plot_flux_analysis":
        pass
    elif pl_name == "do_pathway_analysis":
        pass


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_template(args):
    validate_pipeline_configs("template", args)
    pipeline = TemplatePipeline(TemplatePipelineConfig(
        pipeline=args.pipeline,
        output_path=Path(args.output_path),
    ))
    pipeline.validate()
    if getattr(args, "dry_run", False):
        print(pipeline.plan().format())
        return
    pipeline.run()


def _cmd_integrate(args):
    validate_pipeline_configs("integrate", args)
    pipeline = IntegrationPipeline(IntegrationPipelineConfig.from_files(
        gene_data=args.gene_data_conf_path,
        model=args.model_testing_conf_path,
        threshold=args.threshold_conf_path,
        mapping=args.mapping_conf_path,
        integration=args.integration_conf_path,
    ))
    pipeline.validate()
    if args.dry_run:
        print(pipeline.plan().format())
        return
    pipeline.run()


def _cmd_process(args):
    validate_pipeline_configs("process", args)
    pipeline = ModelProcessingPipeline(ModelProcessingPipelineConfig.from_files(
        model=args.model_testing_conf_path,
    ))
    pipeline.validate()
    if args.dry_run:
        print(pipeline.plan().format())
        return
    pipeline.run()


def _cmd_threshold(args):
    validate_pipeline_configs("threshold", args)
    pipeline = ThresholdPipeline(ThresholdPipelineConfig.from_files(
        gene_data=args.gene_data_conf_path,
        threshold=args.threshold_conf_path,
    ))
    pipeline.validate()
    if args.dry_run:
        print(pipeline.plan().format())
        return
    pipeline.run()


def _cmd_flux(args):
    validate_pipeline_configs("flux", args)
    pipeline = FluxPipeline(FluxPipelineConfig.from_files(
        flux_analysis=args.flux_analysis_conf_path,
        model=args.model_testing_conf_path,
        gene_data=getattr(args, "gene_data_conf_path", None),
        threshold=getattr(args, "threshold_conf_path", None),
        mapping=getattr(args, "mapping_conf_path", None),
        integration=getattr(args, "integration_conf_path", None),
    ))
    pipeline.validate()
    if args.dry_run:
        print(pipeline.plan().format())
        return
    pipeline.run()


def _cmd_compare(args):
    validate_pipeline_configs("compare", args)
    pipeline = ComparisonPipeline(ComparisonPipelineConfig.from_files(
        comparison=args.comparison_conf_path,
    ))
    pipeline.validate()
    if args.dry_run:
        print(pipeline.plan().format())
        return
    pipeline.run()


# ---------------------------------------------------------------------------
# Argument parser builders
# ---------------------------------------------------------------------------

def _add_gene_data_arg(parser):
    parser.add_argument("-g", "--gene_data",
                        dest="gene_data_conf_path",
                        metavar="CONFIG_FILE",
                        default=None,
                        help="Path to gene data config TOML file")


def _add_model_arg(parser):
    parser.add_argument("-t", "--model_testing",
                        dest="model_testing_conf_path",
                        metavar="CONFIG_FILE",
                        default=None,
                        help="Path to model config TOML file")


def _add_threshold_arg(parser):
    parser.add_argument("-r", "--threshold",
                        dest="threshold_conf_path",
                        metavar="CONFIG_FILE",
                        default=None,
                        help="Path to threshold config TOML file")


def _add_mapping_arg(parser):
    parser.add_argument("-m", "--mapping",
                        dest="mapping_conf_path",
                        metavar="CONFIG_FILE",
                        default=None,
                        help="Path to mapping config TOML file")


def _add_integration_arg(parser):
    parser.add_argument("-i", "--integration",
                        dest="integration_conf_path",
                        metavar="CONFIG_FILE",
                        default=None,
                        help="Path to integration config TOML file")


def _add_dry_run_arg(parser):
    parser.add_argument("--dry-run",
                        dest="dry_run",
                        action="store_true",
                        help="Validate configs and print the planned actions without running the pipeline")


def _build_parser():
    """Build the top-level argument parser with subcommands."""
    try:
        pkg_version = version("pipeGEM")
    except PackageNotFoundError:
        pkg_version = "unknown"

    parser = argparse.ArgumentParser(
        prog="pipeGEM",
        description=PROGRAM_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version",
                        version=f"pipeGEM {pkg_version}")

    subparsers = parser.add_subparsers(dest="subcommand")

    # --- template ---
    sp_template = subparsers.add_parser(
        "template", help="Generate template config files",
        description=_docs["template"],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sp_template.add_argument("-p", "--pipeline",
                             dest="pipeline",
                             metavar="PIPELINE_NAME",
                             required=True,
                             help="Pipeline name to generate templates for")
    sp_template.add_argument("-o", "--output",
                             dest="output_path",
                             metavar="OUTPUT_DIR",
                             required=True,
                             help="Output directory for generated config files")
    _add_dry_run_arg(sp_template)
    sp_template.set_defaults(func=_cmd_template)

    # --- integrate ---
    sp_integrate = subparsers.add_parser(
        "integrate", help="Run integration pipeline",
        description=_docs["integration"],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_gene_data_arg(sp_integrate)
    _add_model_arg(sp_integrate)
    _add_threshold_arg(sp_integrate)
    _add_mapping_arg(sp_integrate)
    _add_integration_arg(sp_integrate)
    _add_dry_run_arg(sp_integrate)
    sp_integrate.set_defaults(func=_cmd_integrate)

    # --- process ---
    sp_process = subparsers.add_parser(
        "process", help="Run model processing pipeline",
        description=_docs["model_processing"],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_model_arg(sp_process)
    _add_dry_run_arg(sp_process)
    sp_process.set_defaults(func=_cmd_process)

    # --- threshold ---
    sp_threshold = subparsers.add_parser(
        "threshold", help="Compute expression thresholds",
        description=_docs["get_threshold"],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_gene_data_arg(sp_threshold)
    _add_threshold_arg(sp_threshold)
    _add_dry_run_arg(sp_threshold)
    sp_threshold.set_defaults(func=_cmd_threshold)

    # --- flux ---
    sp_flux = subparsers.add_parser(
        "flux", help="Run flux analysis",
        description=_docs["do_flux_analysis"],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sp_flux.add_argument("-f", "--flux_analysis",
                         dest="flux_analysis_conf_path",
                         metavar="CONFIG_FILE",
                         default=None,
                         help="Path to flux analysis config TOML file")
    _add_model_arg(sp_flux)
    _add_gene_data_arg(sp_flux)
    _add_threshold_arg(sp_flux)
    _add_mapping_arg(sp_flux)
    _add_integration_arg(sp_flux)
    _add_dry_run_arg(sp_flux)
    sp_flux.set_defaults(func=_cmd_flux)

    # --- compare ---
    sp_compare = subparsers.add_parser(
        "compare", help="Compare metabolic models",
        description=_docs["do_model_comparison"],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sp_compare.add_argument("-c", "--comparison",
                            dest="comparison_conf_path",
                            metavar="CONFIG_FILE",
                            default=None,
                            help="Path to comparison config TOML file")
    _add_dry_run_arg(sp_compare)
    sp_compare.set_defaults(func=_cmd_compare)

    return parser


# ---------------------------------------------------------------------------
# Legacy argument parsing (for backward compatibility with -n flag)
# ---------------------------------------------------------------------------

def _is_legacy_invocation(argv):
    """Check if the CLI was invoked with the legacy -n/--name flag."""
    return "-n" in argv or "--name" in argv


def _parse_legacy(argv):
    """Parse arguments using the legacy flat argparse style."""
    parser = argparse.ArgumentParser(prog="pipeGEM")
    parser.add_argument("-n", "--name", dest="pl_name", metavar="pipeline-name")
    parser.add_argument("-g", "--gene_data", dest="gene_data_conf_path", default=None)
    parser.add_argument("-t", "--model_testing", dest="model_testing_conf_path", default=None)
    parser.add_argument("-r", "--threshold", dest="threshold_conf_path", default=None)
    parser.add_argument("-m", "--mapping", dest="mapping_conf_path", default=None)
    parser.add_argument("-i", "--integration", dest="integration_conf_path", default=None)
    parser.add_argument("-o", "--output", dest="output_path", default=None)
    parser.add_argument("-p", "--pipeline", dest="pipeline", default=None)
    parser.add_argument("-c", "--comparison", dest="comparison_conf_path", default=None)
    parser.add_argument("-f", "--flux_analysis", dest="flux_analysis_conf_path", default=None)
    parser.add_argument("-j", "--n_jobs", dest="n_jobs", default=None)
    return parser.parse_args(argv)


def _run_legacy(argv):
    """Handle a legacy-style invocation."""
    from pipeGEM.cli._utils import read_configs

    warnings.warn(
        "The '-n <pipeline>' invocation style is deprecated. "
        "Use subcommands instead, e.g.: pipeGEM integrate -g ... -t ...\n"
        "Run 'pipeGEM --help' for the new usage.",
        FutureWarning,
        stacklevel=2,
    )
    args = _parse_legacy(argv)
    config_dic = read_configs({
        "gene_data_conf": args.gene_data_conf_path,
        "model_conf": args.model_testing_conf_path,
        "threshold_conf": args.threshold_conf_path,
        "mapping_conf": args.mapping_conf_path,
        "integration_conf": args.integration_conf_path,
        "comparison_conf": args.comparison_conf_path,
        "fa_conf": args.flux_analysis_conf_path,
    })
    config_dic.update({"output_path": args.output_path,
                       "pipeline": args.pipeline})
    run_pipeline(pl_name=args.pl_name, **config_dic)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    """Main CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]

    try:
        # Legacy invocation detection
        if _is_legacy_invocation(argv):
            _run_legacy(argv)
            return

        parser = _build_parser()
        args = parser.parse_args(argv)

        if args.subcommand is None:
            parser.print_help()
            sys.exit(0)

        args.func(args)

    except FileNotFoundError as e:
        print(f"Error: file not found: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: missing config key: {e}. Check your TOML files.",
              file=sys.stderr)
        sys.exit(1)
    except CLIError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
