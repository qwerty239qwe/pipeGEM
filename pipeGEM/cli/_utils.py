from typing import Union, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ._io import load_medium, load_threshold_analysis, load_gene_data

from pipeGEM import Model, load_model, Group
from pipeGEM.utils import parse_toml_file, save_toml_file
from pipeGEM.data.data import GeneData, MediumData
from pipeGEM.analysis import consistency_testers
from pipeGEM.analysis.tasks import TaskContainer
from pipeGEM.analysis import TaskAnalysis, LocalThresholdAnalysis, rFASTCORMICSThresholdAnalysis, \
    PercentileThresholdAnalysis, ModelScalingResult


pl_needed_config = {"integration": ["thresholds", "gene_data",
                                    "mapping", "model", "gene_data_integration"],
                    "model_processing": ["model"],
                    "get_threshold": ["thresholds", "gene_data"]}


def generate_template_configs(dest_folder, pl_name):
    (Path(dest_folder) / "configs").mkdir(parents=True)
    saved_conf_dir = Path(dest_folder) / "configs"
    template_folder = Path(__file__).parent / "configs"
    for fn in pl_needed_config[pl_name]:
        if (template_folder / fn).with_suffix(".toml").is_file():
            conf = parse_toml_file((template_folder / fn).with_suffix(".toml"))
            save_toml_file((saved_conf_dir / fn).with_suffix(".toml"), conf)
        elif (template_folder / fn).is_dir():
            (Path(dest_folder) / "configs" / fn).mkdir(parents=True)
            for dfn in (template_folder / fn).iterdir():
                conf = parse_toml_file(dfn)
                save_toml_file((saved_conf_dir / fn / dfn.stem).with_suffix(".toml"), conf)


def read_configs(config_path_dict):
    returned_dic = {}
    for k, v in config_path_dict.items():
        if v is not None:
            returned_dic[k] = parse_toml_file(v)
    return returned_dic


def preprocess_model(model_conf):
    model = Model(model=load_model(model_conf["input"]["input_file_path"]),
                  **model_conf["param"])

    # medium
    medium, apply_medium_first, apply_medium_after = load_medium(model_conf["medium_data"])
    if apply_medium_first:
        model.add_medium_data(model_conf["medium_data"]["name"], medium)
        model.apply_medium(name=model_conf["medium_data"]["name"],
                           **model_conf["medium_data"]["apply_params"])

    # rescale
    if model_conf["rescale"]["precompute_path"] is not None:
        rescaled_result = ModelScalingResult.load(model_conf["rescale"]["precompute_path"])
    else:
        rescaled_result = model.check_model_scale(method=model_conf["rescale"]["method"],
                                                  n_iter=model_conf["rescale"]["n_iter"])
        #Path(model_conf["rescale"]["saved_path"]).mkdir(exist_ok=True)
        rescaled_result.save(model_conf["rescale"]["saved_path"])

    # consistency
    cons_tester = consistency_testers[model_conf["consistency"]["method"]](rescaled_result.rescaled_model)

    if model_conf["consistency"]["method"] == "FASTCC":
        cons_result = cons_tester.analyze(rxn_scaling_coefs=rescaled_result.rxn_scaling_factor,
                                          **model_conf["consistency"]["params"])
    else:
        cons_result = cons_tester.analyze(**model_conf["consistency"]["params"])
    # model.remove_reactions(cons_result.removed_rxn_ids)
    # cons_result.add_result(dict(consistent_model=model))
    #Path(model_conf["consistency"]["saved_path"]).mkdir(exist_ok=True)
    cons_result.save(model_conf["consistency"]["saved_path"])
    model = cons_result.consistent_model

    # metabolic task testing
    ft_params = model_conf["functionality_test"]["params"]
    tt_params = model_conf["functionality_test"]["test_tasks"]
    tasks = TaskContainer.load(ft_params["tasks_file_name"])
    model.add_tasks("default", tasks=tasks)
    task_result = model.test_tasks(name="default",
                                   met_scaling_coefs=rescaled_result.met_scaling_factor,
                                   **tt_params)
    #Path(model_conf["functionality_test"]["saved_path"]).mkdir(exist_ok=True)
    task_result.save(model_conf["functionality_test"]["saved_path"])
    return model, task_result


def find_threshold(gene_data: Union[GeneData, Dict[str, GeneData]],
                   threshold_config):
    th_name = threshold_config["params"]["name"]
    if "plot" in threshold_config and "output_path" in threshold_config["plot"] and \
            threshold_config["plot"]["output_path"] is not None:
        plot_op_path = Path(threshold_config["plot"]["output_path"])
        if not plot_op_path.is_dir():
            plot_op_path.mkdir(parents=True, exist_ok=True)
    else:
        plot_op_path = None

    if th_name != "local":
        result = {}
        for g_name, data in gene_data.items():
            if (Path(threshold_config["saved_path"]) / g_name).is_dir():
                if th_name == "rFASTCORMICS":
                    result[g_name] = rFASTCORMICSThresholdAnalysis.load(
                        str(Path(threshold_config["saved_path"]) / g_name))
                elif th_name == "percentile":
                    result[g_name] = PercentileThresholdAnalysis.load(
                        str(Path(threshold_config["saved_path"]) / g_name))
                continue
            th = data.get_threshold(**threshold_config["params"])
            th.save(Path(threshold_config["saved_path"]) / g_name)
            result[g_name] = th
            if plot_op_path is not None:
                th.plot(file_name=(plot_op_path / g_name).with_suffix(".png"),
                        dpi=450)
    else:
        if Path(threshold_config["saved_path"]).is_dir():
            return LocalThresholdAnalysis.load(threshold_config["saved_path"])

        assert isinstance(gene_data, dict)
        if threshold_config["group"]["group_file_name"] is not None:
            grps = pd.read_csv(threshold_config["group"]["group_file_name"])
            grp_name = threshold_config["group"]["group_name"]
        else:
            grps = None
            grp_name = None

        agg_data = GeneData.aggregate(gene_data, prop="data", group_annotation=grps)
        result = agg_data.find_local_threshold(group_name=grp_name, **threshold_config["params"])
        result.save(threshold_config["saved_path"])
        if plot_op_path is not None:
            result.plot(file_name=(plot_op_path / "local").with_suffix(".png"),
                        dpi=450)

    return result


def map_data(gene_data,
             data_name,
             model,
             mapping_config):
    th_file_name = mapping_config["threshold_analysis"]["input_file_path_pattern"].format(data_name=data_name)
    th_type = mapping_config["threshold_analysis"]["type"]
    threshold_analysis = load_threshold_analysis(th_file_name, th_type)
    exp_th = 0 if th_type == "local" else threshold_analysis.exp_th
    align_kws = mapping_config["rxn_score"]["align"]
    for k, v in align_kws.items():
        if v is None:
            align_kws[k] = np.nan
    model.add_gene_data(name_or_prefix=data_name,
                        data=gene_data,
                        **align_kws)
    if th_type == "local":
        model.gene_data[data_name].assign_local_threshold(threshold_analysis)
    task_analysis = TaskAnalysis.load(mapping_config["task_score"]["input_file_path"])
    task_supp_rxns = model.get_activated_task_sup_rxns(data_name=data_name,
                                                       task_analysis=task_analysis,
                                                       score_threshold=exp_th,
                                                       **mapping_config["task_score"]["get_supp_rxns"])
    return task_supp_rxns


def _integration_get_model_and_task(model_conf, integration_conf):
    model, task_result, rxn_s_factors = None, None, None
    if integration_conf["precompute"]["model"]["cobra_model_path"] is not None:
        prec_mod_params = integration_conf["precompute"]["model"]
        model = Model(model=load_model(prec_mod_params["cobra_model_path"]),
                      name_tag=prec_mod_params["model_name_tag"])
    if integration_conf["precompute"]["tasks"]["task_result_path"] is not None:
        prec_tasks_path = integration_conf["precompute"]["tasks"]["task_result_path"]
        task_result = TaskAnalysis.load(file_path=prec_tasks_path)
    if "rxn_scaling_factor" in integration_conf["precompute"] and \
        (integration_conf["precompute"]["rxn_scaling_factor"]["file_path"] is not None):
        rxn_s_factors = parse_toml_file(integration_conf["precompute"]["rxn_scaling_factor"]["file_path"])
    if model is None and task_result is None:
        model, task_result = preprocess_model(model_conf=model_conf)
    return model, task_result, rxn_s_factors


def _integration_get_thres(gene_data, threshold_config, integration_conf):
    th_path = integration_conf["precompute"]["threshold"]["threshold_result_path"]
    th_type = integration_conf["precompute"]["threshold"]["type"]
    if th_path is not None:
        if th_type == "local":
            return LocalThresholdAnalysis.load(th_path)
        elif th_type == "rFASTCORMICS":
            th_dic = {}
            for data_name in gene_data:
                th_dic[data_name] = rFASTCORMICSThresholdAnalysis.load(Path(th_path) / data_name)
            return th_dic
        elif th_type == "percentile":
            th_dic = {}
            for data_name in gene_data:
                th_dic[data_name] = PercentileThresholdAnalysis.load(Path(th_path) / data_name)
            return th_dic
    return find_threshold(gene_data=gene_data,
                          threshold_config=threshold_config)


def _preprocess_int_configs(integration_conf,
                            th_result,
                            protected_rxns,
                            rxn_s_factors):
    integration_conf = {k: v for k, v in integration_conf.items()}
    int_name = integration_conf.pop("integrator_name")
    integration_conf["integrator"] = int_name
    _ = integration_conf.pop("precompute")
    if int_name == "GIMME":
        if integration_conf["high_exp"] == "default":
            integration_conf["high_exp"] = th_result.exp_th if not isinstance(th_result, LocalThresholdAnalysis) else 0
    elif int_name not in ["RIPTiDeSampling", "EFlux"]:
        integration_conf["predefined_threshold"] = th_result

    integration_conf["protected_rxns"] = integration_conf.get("protected_rxns", []) + protected_rxns
    integration_conf["rxn_scaling_coefs"] = rxn_s_factors
    return integration_conf


def run_integration_pipeline(gene_data_conf,
                             model_conf,
                             threshold_conf,
                             mapping_conf,
                             integration_conf,
                             **kwargs):
    integration_conf = {k: v for k, v in integration_conf.items()}
    gene_data_dic = load_gene_data(gene_data_conf=gene_data_conf)
    model, task_result, rxn_s_factors = _integration_get_model_and_task(model_conf=model_conf,
                                                                        integration_conf=integration_conf)
    th_result = _integration_get_thres(gene_data=gene_data_dic,
                                       threshold_config=threshold_conf,
                                       integration_conf=integration_conf)
    task_supp_rxns = {}
    saved_path = Path(integration_conf.pop("saved_path"))
    for g_name, g_data in gene_data_dic.items():

        if (saved_path / g_name).is_dir():
            print(f"{g_name} is already there. Skipping it")
            continue

        task_supp_rxns[g_name] = map_data(data_name=g_name,
                                          gene_data=g_data,
                                          model=model,
                                          mapping_config=mapping_conf)
        int_c = _preprocess_int_configs(integration_conf=integration_conf,
                                        th_result=th_result[g_name]
                                        if isinstance(th_result, dict) else th_result,
                                        protected_rxns=task_supp_rxns[g_name],
                                        rxn_s_factors=rxn_s_factors)
        int_result = model.integrate_gene_data(data_name=g_name,
                                               **int_c)
        int_result.save(Path(saved_path) / g_name)
        del int_result
        del task_supp_rxns[g_name]


def do_model_comparison(comparison_configs):
    input_path = Path(comparison_configs["models_input_path"])
    model_dic = {}
    for model in input_path.iterdir():
        if comparison_configs["model_type"] == "pg":
            model_dic[model.stem] = Model.load_model(model)
        elif comparison_configs["model_type"] == "cobra":
            model_dic[model.stem] = load_model(str(model))
        else:
            raise ValueError(comparison_configs["model_type"], "must be either pg or cobra.")

    factors = None
    if Path(comparison_configs["factor_file"]).is_file():
        factors = pd.read_csv(comparison_configs["factor_file"],
                              index_col=0)
        print("Factor data loaded, top rows are:")
        print(factors.head())
    grp = Group(model_dic, name_tag="group", factors=factors)
    print(grp.get_info())
    Path(comparison_configs["output_dir"]).mkdir(parents=True, exist_ok=True)
    root = Path(comparison_configs["output_dir"])
    #  number of comp
    num_comp = grp.compare(method="num",
                           group_by=comparison_configs["compare_num"]["group"],)
    num_comp.plot(file_name=root/comparison_configs["compare_num"]["output_file_name"],
                  group=comparison_configs["compare_num"]["group"],
                  dpi=comparison_configs["compare_num"]["dpi"])
    plt.show(block=False)
    plt.close('all')

    #  jaccard
    num_comp = grp.compare(method="jaccard",
                           group_by=None, )
    num_comp.plot(file_name=root/comparison_configs["compare_jaccard"]["output_file_name"],
                  row_color_by=comparison_configs["compare_jaccard"]["row_color_by"],
                  col_color_by=comparison_configs["compare_jaccard"]["col_color_by"],
                  dpi=comparison_configs["compare_jaccard"]["dpi"])
    plt.show(block=False)
    plt.close('all')

    # pca
    num_comp = grp.compare(method="PCA",
                           group_by=None, )
    num_comp.plot(file_name=root/comparison_configs["compare_PCA"]["output_file_name"],
                  dpi=comparison_configs["compare_PCA"]["dpi"],
                  color_by=comparison_configs["compare_PCA"]["color_by"],
                  prefix="")
    plt.show(block=False)
    plt.close('all')


def _load_exist_model(path_pattern, model_name, model_type):
    if model_type == "cobra":
        return Model(model=load_model(path_pattern.format(model_name)), name_tag=model_name)
    if model_type == "pg":
        return Model.load_model(path_pattern.format(model_name))


def _fa_with_data(multi_model_conf,
                  gene_data_conf,
                  threshold_conf,
                  mapping_conf,
                  integration_conf):
    integration_conf = {k: v for k, v in integration_conf.items()}
    model_path_struct = multi_model_conf["models_input_path"]
    model_type = multi_model_conf["model_type"]
    gene_data_dic = load_gene_data(gene_data_conf=gene_data_conf)
    th_result = _integration_get_thres(gene_data=gene_data_dic,
                                       threshold_config=threshold_conf,
                                       integration_conf=integration_conf)
    # prec_tasks_path = integration_conf["precompute"]["tasks"]["task_result_path"]
    # task_result = TaskAnalysis.load(file_path=prec_tasks_path)
    if "rxn_scaling_factor" in integration_conf["precompute"] and \
        (integration_conf["precompute"]["rxn_scaling_factor"]["file_path"] is not None):
        rxn_s_factors = parse_toml_file(integration_conf["precompute"]["rxn_scaling_factor"]["file_path"])
    else:
        rxn_s_factors = None
    task_supp_rxns = {}
    saved_path = integration_conf.pop("saved_path")
    for g_name, g_data in gene_data_dic.items():
        file_saved_path = saved_path.format(g_name)
        if Path(file_saved_path).is_dir():
            continue

        model = _load_exist_model(model_path_struct, g_name, model_type)
        task_supp_rxns[g_name] = map_data(data_name=g_name,
                                          gene_data=g_data,
                                          model=model,
                                          mapping_config=mapping_conf)
        int_c = _preprocess_int_configs(integration_conf=integration_conf,
                                        th_result=th_result[g_name]
                                        if isinstance(th_result, dict) else th_result,
                                        protected_rxns=task_supp_rxns[g_name],
                                        rxn_s_factors=rxn_s_factors)

        int_result = model.integrate_gene_data(data_name=g_name,
                                               **int_c)
        if not Path(file_saved_path).parent.is_dir():
            Path(file_saved_path).parent.mkdir(parents=True, exist_ok=True)

        int_result.save(file_saved_path)


def do_flux_analysis(fa_configs,
                     multi_model_conf,
                     gene_data_conf,
                     threshold_conf,
                     mapping_conf,
                     integration_conf,
                     ):
    if integration_conf is not None:
        _fa_with_data(multi_model_conf, gene_data_conf, threshold_conf, mapping_conf, integration_conf)
        return
    else:
        model_path_dir = multi_model_conf["models_input_dir"]
        model_type = multi_model_conf["model_type"]
        group_factor = pd.read_csv(multi_model_conf["group_factor_path"])
        models = []
        for fn in Path(model_path_dir).iterdir():
            if model_type == "cobra":
                models.append(Model(model=load_model(str(fn)), name_tag=fn.stem))
            elif model_type == "pg":
                models.append(Model.load_model(fn))

        group = Group(group=models, name_tag="group", factors=group_factor)

        fa_saved_path = fa_configs.pop("saved_path")
        flux_result = group.do_flux_analysis(**fa_configs)
        flux_result.save(fa_saved_path)

