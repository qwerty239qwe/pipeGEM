from pathlib import Path
from os import PathLike
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import cobra
import pandas as pd
from biodbs.BioMart import Dataset


CORDA_THRESHOLDS = {"discrete": {"HC": (np.inf, 3), "MC": (2, 1), "NC": (-1, -np.inf)},
                    "continuous": {"HC": (np.inf, 2.5), "MC": (2.5, 1), "NC": (-0.5, -np.inf)}}
HPA_SCORE_COLS = ["pTPM", "NX"]


def get_gene_id_map(gene_names: List[str],
                    from_id: str,
                    to_id: str,
                    df_path: Union[PathLike, str],
                    ds_kws: Dict[str, Any],
                    map_type: str = "df",
                    drop_unused: bool = False,
                    ref_model: Optional[cobra.Model] = None):
    """
    Get a gene ID mapper from local path or Biomart

    Parameters
    ----------
    gene_names: list of str
        The gene names / IDs to be translate into another gene names or IDs
    from_id: str
        The name of the current IDs
    to_id: str
        The name of the transformed IDs
    df_path: pathlike or str

    ds_kws
    map_type
    drop_unused
    ref_model

    Returns
    -------

    """
    assert drop_unused == (ref_model is not None), "ref_model should be assigned when drop_unused is True"
    if df_path is None:
        ds = Dataset(**ds_kws)
        filter_kwargs = {from_id: gene_names}
        map_df = ds.get_data(attribs=[to_id, from_id], **filter_kwargs)
    else:
        df_path = Path(df_path)
        if not df_path.is_file():  # save df at the given path
            ds = Dataset(**ds_kws)
            filter_kwargs = {from_id: gene_names}
            map_df = ds.get_data(attribs=[to_id, from_id], **filter_kwargs)
            map_df.to_csv(df_path, sep='\t')  # save df
        map_df = pd.read_csv(df_path, sep='\t', dtype=str)
    if drop_unused:
        relevant_genes = [g.id for g in ref_model.genes]
        map_df = map_df[map_df[to_id].isin(relevant_genes)]
    assert to_id in map_df.columns and from_id in map_df.columns, map_df.columns
    if map_type == "dict":
        map_df = {f: t for f, t in zip(map_df[from_id], map_df[to_id])}
    return {"map_df": map_df}


def translate_gene_id(data_df: pd.DataFrame,
                      map_df: pd.DataFrame,
                      gene_col: str,
                      to_id: str):
    data_df[to_id] = data_df[gene_col].map(map_df, na_action="") \
        if gene_col != "index" else data_df.index.map(map_df, na_action="")

    data_df = data_df[data_df[to_id] != ""]
    if to_id == "index":
        data_df.index = data_df[to_id]
        data_df = data_df.drop(columns=[to_id])
    return {"data_df": data_df, "gene_id_col": to_id}


def unify_score_column(data_df: pd.DataFrame,
                       level_dic: Dict[str, float],
                       score_col_name: str) -> (pd.DataFrame, Dict[str, Dict[str, Tuple[float, float]]]):
    # unify value column (for pathology data)

    data_df = data_df.copy()
    score_series, total_series = pd.Series({}, index=data_df.index).fillna(0), \
                                 pd.Series({}, index=data_df.index).fillna(0)
    level_cols = []
    for level, val in level_dic.items():
        if level in data_df.columns:
            level_cols.append(level)
            score_series += (data_df[level] * val)
            total_series += data_df[level]
    print(level_cols)
    if len(level_cols) != 0:
        print("Using thresholds for continuous data")
        used_rxn_thres = CORDA_THRESHOLDS["continuous"]
        score_series /= total_series
        data_df = data_df.drop(level_cols)
        data_df["score"] = score_series
    elif "Level" in data_df.columns:
        print("Using thresholds for discrete data")
        used_rxn_thres = CORDA_THRESHOLDS["discrete"]
        data_df["Level"] = data_df["Level"].apply(lambda x: level_dic[x] if x in level_dic else 0)
        data_df.rename(columns={"Level": "score"}, inplace=True)
    else:
        # TODO: use fastcormic thresholding
        print("Using Fastcormic thresholds")
        used_rxn_thres = None
        for score_col in HPA_SCORE_COLS:
            if score_col in data_df.columns:
                data_df.rename(columns={score_col: score_col_name}, inplace=True)
                break
    return {"data_df": data_df, "used_rxn_thres": used_rxn_thres, "score_col_name": score_col_name}


def transform_HPA_data(data_df,
                       categories: List[str],
                       gene_id_col: str = "entrezgene",
                       score_col_name: str = "score"):
    # groupby (merge data in interested cols)
    data_df = data_df.copy()
    print(data_df)
    if gene_id_col == "index":
        data_df["index_"] = data_df.index
        gene_id_col = "index_"
    data_df = data_df.reindex(columns=[gene_id_col, score_col_name] + categories).groupby([gene_id_col] + categories).mean()
    data_df = data_df.reset_index()
    sample_cols = [col for col in data_df if col not in [gene_id_col, score_col_name]]
    if len(sample_cols) > 1:
        # sample multiindex
        data_df["sample"] = data_df.apply(lambda x: "_".join([x[c] for c in sample_cols]))
        data_df.drop(sample_cols)
    elif len(sample_cols) == 1:
        data_df.rename(columns={sample_cols[0]: "sample"}, inplace=True)
    else:
        raise ValueError("data_df should contain at least one sample col")

    sample_df = data_df.pivot(index=gene_id_col, columns="sample", values=score_col_name)
    return {"data_df": sample_df}


def get_uniform_expr_threshold_dic(data_df,
                                   sample_names,
                                   expr_threshold,
                                   non_expr_threshold
                                   ):
    valid_sample_names = data_df.columns.to_list()
    if sample_names is None:
        sample_names = valid_sample_names
    else:
        sample_names = list(set(valid_sample_names) & set(sample_names))

    expr_threshold_dic = {s: expr_threshold for s in sample_names}
    non_expr_threshold_dic = {s: non_expr_threshold for s in sample_names}
    return {"_sample_names": sample_names,
            "_expr_threshold_dic": expr_threshold_dic,
            "_non_expr_threshold_dic": non_expr_threshold_dic}
