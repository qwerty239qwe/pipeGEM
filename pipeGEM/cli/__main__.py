import argparse
from pathlib import Path
from typing import Union, Dict

from ._io import load_medium, load_threshold_analysis, load_gene_data
from ._doc import get_help_doc
from ._utils import preprocess_model, find_threshold, map_data, \
    integrate_gene_data, read_configs, generate_template_configs

from pipeGEM.utils import parse_toml_file


def run_integration_pipeline(gene_data_conf,
                             model_conf,
                             threshold_conf,
                             mapping_conf,
                             **kwargs):
    gene_data_dic = load_gene_data(gene_data_conf=gene_data_conf)
    model, task_result = preprocess_model(model_conf=model_conf)
    th_result = find_threshold(gene_data=gene_data_dic,
                               threshold_config=threshold_conf)
    task_supp_rxns = {}
    for g_name, g_data in gene_data_dic.items():
        task_supp_rxns[g_name] = map_data(data_name=g_name,
                                          gene_data=g_data,
                                          model=model,
                                          mapping_config=mapping_conf)


# TODO: do_flux_analysis, do_model_comparison, do_pathway_analysis
def main(pl_name, **configs):
    if pl_name == "template":
        generate_template_configs(dest_folder=configs.get("output_path"),
                                  pl_name=configs.get("pipeline"))

    if pl_name == "integration":
        run_integration_pipeline(**configs)
    if pl_name == "model_processing":
        _, _ = preprocess_model(model_conf=configs.get("model_conf"))
    if pl_name == "get_threshold":
        gene_data_dic = load_gene_data(gene_data_conf=configs.get("gene_data_conf"))
        _ = find_threshold(gene_data_dic, configs.get("threshold_conf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name",
                        dest="pl_name",
                        metavar="pipeline-name",
                        help=get_help_doc("all"))

    parser.add_argument("-g", "--gene_data",
                        dest="gene_data_conf_path",
                        metavar="gene_data_config_file_path",
                        default=None)

    parser.add_argument("-t", "--model_testing",
                        dest="model_testing_conf_path",
                        metavar="model_testing_config_file_path",
                        default=None)

    parser.add_argument("-r", "--threshold",
                        dest="threshold_conf_path",
                        metavar="threshold_config_file_path",
                        default=None)

    parser.add_argument("-m", "--mapping",
                        dest="mapping_conf_path",
                        metavar="mapping_config_file_path",
                        default=None)

    parser.add_argument("-o", "--output",
                        dest="output_path",
                        metavar="output_path",
                        default=None)

    parser.add_argument("-p", "--pipeline",
                        dest="pipeline",
                        metavar="pipeline-name",
                        default=None)

    args = parser.parse_args()
    config_dic = read_configs({"gene_data_conf": args.gene_data_conf_path,
                               "model_conf": args.model_testing_conf_path,
                               "threshold_conf": args.threshold_conf_path,
                               "mapping_conf": args.mapping_conf_path})
    config_dic.update({"output_path": args.output_path,
                       "pipeline": args.pipeline})
    main(pl_name=args.pl_name, **config_dic)