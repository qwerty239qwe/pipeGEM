from typing import Union, Dict
from pathlib import Path

from ._io import load_medium, load_threshold_analysis, load_gene_data

from pipeGEM import Model, load_model
from pipeGEM.utils import parse_toml_file, save_toml_file
from pipeGEM.data.data import GeneData, MediumData
from pipeGEM.analysis import consistency_testers
from pipeGEM.analysis.tasks import TaskContainer
from pipeGEM.analysis import TaskAnalysis


pl_needed_config = {"integration": ["thresholds", "gene_data",
                                    "mapping", "model", "gene_data_integration"],
                    "model_processing": ["model"],
                    "get_threshold": ["thresholds"]}


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

    # consistency
    cons_tester = consistency_testers[model_conf["consistency"]["method"]](model)
    cons_result = cons_tester.analyze(**model_conf["consistency"]["params"])
    cons_result.save(model_conf["consistency"]["saved_path"])
    model = cons_result.consistent_model

    # metabolic task testing
    ft_params = model_conf["functionality_test"]["params"]
    tt_params = model_conf["functionality_test"]["test_tasks"]
    tasks = TaskContainer.load(ft_params["tasks_file_name"])
    model.add_tasks("default", tasks=tasks)
    task_result = model.test_tasks(name="default",
                                   **tt_params)
    task_result.save(model_conf["functionality_test"]["saved_path"])
    return model, task_result


def find_threshold(gene_data: Union[GeneData, Dict[str, GeneData]],
                   threshold_config):
    th_name = threshold_config["params"]["name"]
    if th_name != "local":
        result = gene_data.get_threshold(**threshold_config["params"])
    else:
        assert isinstance(gene_data, dict)
        agg_data = GeneData.aggregate(gene_data, prop="data")
        result = agg_data.find_local_threshold(**threshold_config["params"])
    result.save(threshold_config["saved_path"])
    return result


def map_data(gene_data,
             data_name,
             model,
             mapping_config):
    th_file_name = mapping_config["threshold_analysis"]["input_file_path"].format(data_name=data_name)
    th_type = mapping_config["threshold_analysis"]["type"]
    threshold_analysis = load_threshold_analysis(th_file_name, th_type)
    exp_th = 0 if th_type == "local" else threshold_analysis.exp_th

    model.add_gene_data(name_or_prefix=data_name,
                        data=gene_data,
                        **mapping_config["rxn_score"]["align"])
    task_analysis = TaskAnalysis.load(mapping_config["task_score"]["input_file_path"])
    task_supp_rxns = model.get_activated_task_sup_rxns(data_name=data_name,
                                                       task_analysis=task_analysis,
                                                       score_threshold=exp_th,
                                                       **mapping_config["task_score"]["get_supp_rxns"])
    return gene_data.rxn_scores, task_supp_rxns


def integrate_gene_data(model,
                        data_name,
                        th_analysis,
                        task_supp_rxns,
                        integration_configs: dict):
    integrator_name = integration_configs.pop("integrator_name")
    protected_rxns = integration_configs.pop("protected_rxns", [])
    result = model.integrate_gene_data(data_name=data_name,
                                       integrator=integrator_name,
                                       protected_rxns=protected_rxns + task_supp_rxns,
                                       **integration_configs)
    result.save()
    return result