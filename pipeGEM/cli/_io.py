import pandas as pd
import tomlkit
import numpy as np
from pathlib import Path
import scanpy as sc
import anndata as ad

from pipeGEM.data.data import GeneData, MediumData
from pipeGEM.analysis import rFASTCORMICSThresholdAnalysis, LocalThresholdAnalysis, PercentileThresholdAnalysis


dense_data_loader = {"csv": pd.read_csv,
                     "tsv": pd.read_csv,
                     "txt": pd.read_csv,
                     "xlsx": pd.read_excel,
                     "json": pd.read_json,
                     "parquet": pd.read_parquet,
                     "fwf": pd.read_fwf}
sparse_data_loader = {"h5ad": sc.read_h5ad,
                      "txt": sc.read_text}


def load_data(file_path,
              data_type='infer',
              parser="csv",
              **kwargs):
    file_path = Path(file_path)

    if data_type == 'infer':
        if file_path.suffix[1:] in dense_data_loader:
            return dense_data_loader[file_path.suffix[1:]](file_path, **kwargs)
        elif file_path.suffix[1:] in sparse_data_loader:
            return sparse_data_loader[file_path.suffix[1:]](file_path, **kwargs)
        else:
            raise ValueError(f"cannot infer file with suffix: {file_path.suffix},"
                       f"please manually select data_type and parser")
    elif data_type == "dense":
        return dense_data_loader[parser](file_path, **kwargs)
    elif data_type == "sparse":
        return sparse_data_loader[parser](file_path, **kwargs)
    raise ValueError(r"Please select data_type from 'infer', 'dense', and 'sparse'")


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


def load_threshold_analysis(input_file_path, th_type):
    analysis_dic = {"rFASTCORMICS": rFASTCORMICSThresholdAnalysis,
                    "local": LocalThresholdAnalysis,
                    "percentile": PercentileThresholdAnalysis}
    return analysis_dic[th_type].load(input_file_path)