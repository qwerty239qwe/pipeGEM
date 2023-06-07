import argparse
from pathlib import Path

import pandas as pd
import anndata as ad

from ._io import load_data

from pipeGEM import Model, load_model
from pipeGEM.utils import parse_toml_file
from pipeGEM.data.data import GeneData, MediumData
from pipeGEM.analysis import consistency_testers
from pipeGEM.analysis.tasks import TaskContainer, TaskHandler


def load_gene_data(gene_data_conf):
    inp_params = gene_data_conf["input"]
    input_file_path = inp_params.pop("input_file_path")
    data = load_data(file_path=input_file_path, **inp_params)
    gene_data_dic = {}
    if isinstance(data, pd.DataFrame):
        for name in data.columns:
            gene_data_dic[name] = GeneData(data=data.loc[:, name], **gene_data_conf["params"])
    elif isinstance(data, ad.AnnData):
        for name in data.obs.index:
            gene_data_dic[name] = GeneData(data=data[name, :], **gene_data_conf["params"])

    return gene_data_dic


def load_medium(medium_data_conf):
    apply_medium_first = medium_data_conf["apply_before"]
    apply_medium_after = medium_data_conf["apply_after"]

    if apply_medium_first or apply_medium_after:
        medium_kws = medium_data_conf["params"]
        medium_fp = medium_kws.pop("file_path")
        medium = MediumData.from_file(file_name=medium_fp, **medium_kws)
    else:
        medium = None
    return medium, apply_medium_first, apply_medium_after


def model_preprocess(model_conf):
    model = Model(model=load_model(model_conf["input_file_path"]), **model_conf["param"])

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


def main():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--gene_data",
                        dest="gene_data_conf_path",
                        metavar="config_file_path")

    main()