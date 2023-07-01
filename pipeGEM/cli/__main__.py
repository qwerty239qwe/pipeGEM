import argparse
from pathlib import Path
from typing import Union, Dict

from pipeGEM.utils import parse_toml_file
from pipeGEM.cli._io import load_medium, load_threshold_analysis, load_gene_data
from pipeGEM.cli._doc import get_help_doc
from pipeGEM.cli._utils import preprocess_model, find_threshold, map_data, \
    read_configs, generate_template_configs, run_integration_pipeline, do_model_comparison, \
    do_flux_analysis


# TODO: do_flux_analysis, do_pathway_analysis
def main(pl_name, **configs):
    if pl_name == "template":
        generate_template_configs(dest_folder=configs.get("output_path"),
                                  pl_name=configs.get("pipeline"))
    if pl_name == "integration":
        run_integration_pipeline(gene_data_conf=configs.get("gene_data_conf"),
                                 model_conf=configs.get("model_conf"),
                                 threshold_conf=configs.get("threshold_conf"),
                                 mapping_conf=configs.get("mapping_conf"),
                                 integration_conf=configs.get("integration_conf"))
    if pl_name == "model_processing":
        _, _ = preprocess_model(model_conf=configs.get("model_conf"))
    if pl_name == "get_threshold":
        gene_data_dic = load_gene_data(gene_data_conf=configs.get("gene_data_conf"))
        _ = find_threshold(gene_data_dic, configs.get("threshold_conf"))
    if pl_name == "do_flux_analysis":
        do_flux_analysis(fa_configs=configs.get("fa_conf"),
                         multi_model_conf=configs.get("model_conf"),
                         gene_data_conf=configs.get("gene_data_conf"),
                         threshold_conf=configs.get("threshold_conf"),
                         mapping_conf=configs.get("mapping_conf"),
                         integration_conf=configs.get("integration_conf"))
    if pl_name == "do_model_comparison":
        do_model_comparison(comparison_configs=configs.get("comparison_conf"))
    if pl_name == "plot_flux_analysis":
        pass
    if pl_name == "do_pathway_analysis":
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name",
                        dest="pl_name",
                        metavar="pipeline-name",
                        help=get_help_doc("all"))

    parser.add_argument("-g", "--gene_data",
                        dest="gene_data_conf_path",
                        metavar="config_file_path",
                        default=None)

    parser.add_argument("-t", "--model_testing",
                        dest="model_testing_conf_path",
                        metavar="config_file_path",
                        default=None)

    parser.add_argument("-r", "--threshold",
                        dest="threshold_conf_path",
                        metavar="config_file_path",
                        default=None)

    parser.add_argument("-m", "--mapping",
                        dest="mapping_conf_path",
                        metavar="config_file_path",
                        default=None)

    parser.add_argument("-i", "--integration",
                        dest="integration_conf_path",
                        metavar="config_file_path",
                        default=None)

    parser.add_argument("-o", "--output",
                        dest="output_path",
                        metavar="output_path",
                        default=None)

    parser.add_argument("-p", "--pipeline",
                        dest="pipeline",
                        metavar="pipeline-name",
                        default=None)

    parser.add_argument("-c", "--comparison",
                        dest="comparison_conf_path",
                        metavar="config_file_path",
                        default=None)

    parser.add_argument("-f", "--flux_analysis",
                        dest="flux_analysis_conf_path",
                        metavar="config_file_path",
                        default=None)

    parser.add_argument("-j", "--n_jobs",
                        dest="n_jobs",
                        metavar="n_workers",
                        default=None)

    args = parser.parse_args()
    config_dic = read_configs({"gene_data_conf": args.gene_data_conf_path,
                               "model_conf": args.model_testing_conf_path,
                               "threshold_conf": args.threshold_conf_path,
                               "mapping_conf": args.mapping_conf_path,
                               "integration_conf": args.integration_conf_path,
                               "comparison_conf": args.comparison_conf_path,
                               "fa_conf": args.flux_analysis_conf_path})
    config_dic.update({"output_path": args.output_path,
                       "pipeline": args.pipeline})
    main(pl_name=args.pl_name, **config_dic)