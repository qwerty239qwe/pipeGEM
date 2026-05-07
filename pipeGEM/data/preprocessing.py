from pathlib import Path
from os import PathLike
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import cobra
import pandas as pd
from biodbs.fetch import biomart_query

from pipeGEM._logging import get_logger

logger = get_logger(__name__)


CORDA_THRESHOLDS = {"discrete": {"HC": (np.inf, 3), "MC": (2, 1), "NC": (-1, -np.inf)},
                    "continuous": {"HC": (np.inf, 2.5), "MC": (2.5, 1), "NC": (-0.5, -np.inf)}}
HPA_SCORE_COLS = ["pTPM", "NX"]


def get_gene_id_map(gene_names: List[str],
                    from_id: str,
                    to_id: str,
                    df_path: Union[PathLike, str],
                    dataset: Union[str, Dict] = "hsapiens_gene_ensembl",
                    ds_kws: Optional[Dict] = None,
                    map_type: str = "df",
                    drop_unused: bool = False,
                    ref_model: Optional[cobra.Model] = None):
    """
    Get a gene ID mapper from local path or BioMart.

    Parameters
    ----------
    gene_names : list of str
        The gene names / IDs to be translated into another gene names or IDs.
    from_id : str
        The name of the current IDs (e.g. ``"ensembl_gene_id"``).
    to_id : str
        The name of the transformed IDs (e.g. ``"external_gene_name"``).
    df_path : path-like or str
        Path to cache the mapping DataFrame as a TSV file.  If ``None``,
        the mapping is fetched from BioMart without caching.
    dataset : str
        BioMart dataset name (default ``"hsapiens_gene_ensembl"``).
    ds_kws : dict, optional
        Backward-compatible dataset keyword dictionary used by earlier
        versions. If supplied, ``name``, ``dataset``, or ``dataset_name`` is
        used as the BioMart dataset name.
    map_type : str
        ``"df"`` to return a DataFrame, ``"dict"`` to return a dict.
    drop_unused : bool
        If ``True``, drop genes not present in *ref_model*.
    ref_model : cobra.Model, optional
        Reference model used when *drop_unused* is ``True``.

    Returns
    -------
    dict
        ``{"map_df": ...}`` where the value is a DataFrame or dict.
    """
    if isinstance(dataset, dict) and ds_kws is None:
        ds_kws = dataset
        dataset = "hsapiens_gene_ensembl"
    if ds_kws is not None:
        dataset = ds_kws.get("dataset", ds_kws.get("dataset_name", ds_kws.get("name", dataset)))

    assert drop_unused == (ref_model is not None), "ref_model should be assigned when drop_unused is True"
    if df_path is None:
        result = biomart_query(
            dataset=dataset,
            attributes=[to_id, from_id],
            filters={from_id: gene_names},
        )
        map_df = result.as_dataframe()
    else:
        df_path = Path(df_path)
        if not df_path.is_file():  # save df at the given path
            result = biomart_query(
                dataset=dataset,
                attributes=[to_id, from_id],
                filters={from_id: gene_names},
            )
            map_df = result.as_dataframe()
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
    """Translate gene identifiers in a DataFrame using a mapping.

    Adds a new column (or replaces the index) with the translated IDs
    and drops rows that could not be mapped.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing the original gene identifiers.
    map_df : pd.DataFrame or dict
        Mapping from original IDs to target IDs.  If a DataFrame, it is
        used via :meth:`pandas.Series.map`; if a dict, keys are original
        IDs and values are translated IDs.
    gene_col : str
        Column in *data_df* that holds the source gene IDs.  Use
        ``"index"`` to translate the DataFrame index instead.
    to_id : str
        Name for the new translated-ID column.  If ``"index"``, the
        translated IDs replace the DataFrame index.

    Returns
    -------
    dict
        ``{"data_df": pd.DataFrame, "gene_id_col": str}`` — the
        updated DataFrame and the name of the translated-ID column.
    """
    data_df[to_id] = data_df[gene_col].map(map_df, na_action=None) \
        if gene_col != "index" else data_df.index.map(map_df, na_action=None)

    data_df = data_df[data_df[to_id].notna() & (data_df[to_id] != "")]
    if to_id == "index":
        data_df.index = data_df[to_id]
        data_df = data_df.drop(columns=[to_id])
    return {"data_df": data_df, "gene_id_col": to_id}


def unify_score_column(data_df: pd.DataFrame,
                       level_dic: Dict[str, float],
                       score_col_name: str) -> (pd.DataFrame, Dict[str, Dict[str, Tuple[float, float]]]):
    """Convert heterogeneous HPA expression columns into a single score.

    Handles three cases depending on which columns are present:

    1. **Level count columns** (e.g. ``"High"``, ``"Medium"``, ``"Low"``):
       compute a weighted average using *level_dic* as weights and return
       continuous CORDA thresholds.
    2. **A ``"Level"`` column** with discrete labels: map labels to numeric
       scores via *level_dic* and return discrete CORDA thresholds.
    3. **Quantitative columns** (``"pTPM"`` or ``"NX"``): rename the first
       matching column to *score_col_name* and return ``None`` thresholds
       (thresholding left to downstream methods).

    Parameters
    ----------
    data_df : pd.DataFrame
        HPA expression DataFrame (one row per gene × tissue/cell-type).
    level_dic : dict[str, float]
        Mapping from expression-level labels (e.g. ``"High"``) to numeric
        weights.  Also used to map discrete ``"Level"`` labels.
    score_col_name : str
        Name for the unified score column added to the returned DataFrame.

    Returns
    -------
    dict
        ``{"data_df": pd.DataFrame, "used_rxn_thres": dict or None}`` —
        the updated DataFrame and the CORDA threshold dictionary (or
        ``None`` when quantitative data is used).
    """
    data_df = data_df.copy()
    score_series, total_series = pd.Series({}, index=data_df.index).fillna(0), \
                                 pd.Series({}, index=data_df.index).fillna(0)
    level_cols = []
    for level, val in level_dic.items():
        if level in data_df.columns:
            level_cols.append(level)
            score_series += (data_df[level] * val)
            total_series += data_df[level]
    logger.debug("level_cols: %s", level_cols)
    if len(level_cols) != 0:
        logger.info("Using thresholds for continuous data")
        used_rxn_thres = CORDA_THRESHOLDS["continuous"]
        score_series /= total_series
        data_df = data_df.drop(columns=level_cols)
        data_df["score"] = score_series
    elif "Level" in data_df.columns:
        logger.info("Using thresholds for discrete data")
        used_rxn_thres = CORDA_THRESHOLDS["discrete"]
        data_df["Level"] = data_df["Level"].apply(lambda x: level_dic[x] if x in level_dic else 0)
        data_df.rename(columns={"Level": "score"}, inplace=True)
    else:
        # TODO: use fastcormic thresholding
        logger.info("Using Fastcormic thresholds")
        used_rxn_thres = None
        for score_col in HPA_SCORE_COLS:
            if score_col in data_df.columns:
                data_df.rename(columns={score_col: score_col_name}, inplace=True)
                break
    return {"data_df": data_df,
            "used_rxn_thres": used_rxn_thres}


def transform_HPA_data(data_df,
                       categories: List[str],
                       gene_id_col: str = "entrezgene",
                       score_col_name: str = "score"):
    """Pivot HPA data into a gene × sample expression matrix.

    Groups rows by *gene_id_col* and the specified *categories*, averages
    duplicate entries, then pivots so that each unique combination of
    category values becomes a column (sample).

    Parameters
    ----------
    data_df : pd.DataFrame
        Filtered HPA DataFrame (e.g. output of :func:`unify_score_column`).
    categories : list of str
        Column names that together define a "sample" (e.g.
        ``["Tissue", "Cell type"]``).  Multiple columns are joined with
        ``"_"`` to form a single sample label.
    gene_id_col : str, optional
        Column (or ``"index"``) holding gene identifiers
        (default ``"entrezgene"``).
    score_col_name : str, optional
        Column with numeric expression scores (default ``"score"``).

    Returns
    -------
    dict
        ``{"data_df": pd.DataFrame}`` — a genes × samples matrix where
        rows are genes and columns are sample labels.

    Raises
    ------
    ValueError
        If *categories* is empty (at least one sample column is required).
    """
    data_df = data_df.copy()
    if gene_id_col == "index":
        data_df["index_"] = data_df.index
        gene_id_col = "index_"
    data_df = data_df.reindex(columns=[gene_id_col, score_col_name] + categories).groupby([gene_id_col] + categories).mean()
    data_df = data_df.reset_index()
    sample_cols = [col for col in data_df if col not in [gene_id_col, score_col_name]]
    if len(sample_cols) > 1:
        # sample multiindex
        data_df["sample"] = data_df.apply(lambda x: "_".join([x[c] for c in sample_cols]), axis=1)
        data_df = data_df.drop(columns=sample_cols)
    elif len(sample_cols) == 1:
        data_df.rename(columns={sample_cols[0]: "sample"}, inplace=True)
    else:
        raise ValueError("data_df should contain at least one sample col")

    sample_df = data_df.pivot(index=gene_id_col, columns="sample", values=score_col_name)
    return {"data_df": sample_df}

