from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, fisher_exact

from pipeGEM._logging import get_logger
from pipeGEM.analysis._utils import bh_adjust

logger = get_logger(__name__)


class HyperGeometricTester:
    def __init__(self):
        pass


def hypergeometric_test(data: pd.DataFrame,
                        pathway_col: str,
                        sig_col: str) -> pd.DataFrame:
    """
    Perform hypergeometric test on the given data.

    Parameters
    ----------
    data: pd.DataFrame
        A pandas DataFrame containing the data used for the test.
        It should have two columns: pathway_col indicating the pathway each reaction is categorized into,
        and sig_col, a boolean column indicating whether the differential test of the reaction is significant (True)
        or not (False).
    pathway_col: str
        A string specifying the name of the column in the DataFrame that indicates the pathway each reaction belongs to.
    sig_col: str
        A string specifying the name of the boolean column in the DataFrame that indicates
        whether a particular reaction is significant (True) or not (False) in the differential test.

    Returns
    -------
    result_df: pd.DataFrame
    The function returns a pandas DataFrame named result_df, which contains the following columns:
    `pval`: The raw p-values of the hypergeometric tests for each pathway.
    `padj`: The Benjamini-Hochberg (BH)-adjusted p-values of the hypergeometric tests for each pathway.
        The BH adjustment is a method to control the false discovery rate (FDR).
    `BgRatio`: The ratio of the number of reactions in a specific pathway to the total number of reactions in the dataset.
        This indicates the proportion of reactions in a pathway relative to the whole dataset.
    `SigRatio`: The ratio of the number of significant reactions in a specific pathway
        to the total number of significant reactions in the dataset.
        This shows the proportion of significant reactions in a pathway relative to the total number of significant reactions.
    """
    n_sigs = data[sig_col].astype(int).sum()
    n_pop = data.shape[0]
    cnt_pop_pathway = data.groupby(pathway_col).count().iloc[:, 0].to_dict()
    cnt_subpop_pathway = data[data[sig_col]].groupby(pathway_col).count().iloc[:, 0].to_dict()
    bg_ratio, sig_ratio, p_vals = {}, {}, {}

    for pathway in data[pathway_col].unique():
        n_subpop = cnt_pop_pathway[pathway]
        rv = hypergeom(n_pop, n_sigs, n_subpop)
        bg_ratio[pathway] = f"{n_subpop}/{n_pop}"
        if pathway in cnt_subpop_pathway:
            p_vals[pathway] = rv.sf(cnt_subpop_pathway[pathway] - 1)
            sig_ratio[pathway] = f"{cnt_subpop_pathway[pathway]}/{n_sigs}"
        else:
            p_vals[pathway] = 1
            sig_ratio[pathway] = f"0/{n_sigs}"
    p_vals = pd.Series(p_vals)
    result_df = pd.DataFrame({"pval": p_vals,
                              "padj": pd.Series(bh_adjust(p_vals),
                                                index=p_vals.index),
                              "BgRatio": bg_ratio,
                              "SigRatio": sig_ratio})

    return result_df


def fishers_exact_test(
    data: pd.DataFrame,
    pathway_col: str,
    sig_col: str,
) -> pd.DataFrame:
    """Fisher's exact test for pathway enrichment.

    Parameters
    ----------
    data : pandas.DataFrame
        Same format as :func:`hypergeometric_test`.
    pathway_col, sig_col : str
        Column names.

    Returns
    -------
    pandas.DataFrame
        With columns ``pval``, ``padj``, ``odds_ratio``.
    """
    n_pop = data.shape[0]
    n_sig = data[sig_col].astype(int).sum()
    p_vals, odds = {}, {}

    for pathway in data[pathway_col].unique():
        in_pathway = data[pathway_col] == pathway
        a = (data[sig_col] & in_pathway).sum()   # sig & in pathway
        b = (data[sig_col] & ~in_pathway).sum()  # sig & not in pathway
        c = (~data[sig_col] & in_pathway).sum()  # not sig & in pathway
        d = (~data[sig_col] & ~in_pathway).sum() # not sig & not in pathway
        table = [[a, b], [c, d]]
        or_val, p = fisher_exact(table, alternative="greater")
        p_vals[pathway] = p
        odds[pathway] = or_val

    p_series = pd.Series(p_vals)
    return pd.DataFrame({
        "pval": p_series,
        "padj": pd.Series(bh_adjust(p_series), index=p_series.index),
        "odds_ratio": odds,
    })


def gsea_style_enrichment(
    scores: Dict[str, float],
    pathway_definitions: Dict[str, List[str]],
    n_permutations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Gene Set Enrichment Analysis (GSEA) style ranking-based enrichment.

    Uses ranked reaction flux scores to compute an enrichment score per
    pathway, with permutation-based p-values.

    Parameters
    ----------
    scores : dict
        Mapping of reaction IDs to numeric scores (e.g. flux values or
        fold changes).
    pathway_definitions : dict
        Mapping of pathway names to lists of reaction IDs.
    n_permutations : int
        Number of random permutations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        With columns ``ES`` (enrichment score), ``NES`` (normalised),
        ``pval``, ``padj``.
    """
    rng = np.random.RandomState(seed)
    all_ids = list(scores.keys())
    sorted_ids = sorted(all_ids, key=lambda x: abs(scores.get(x, 0)), reverse=True)
    n = len(sorted_ids)

    def _running_es(ranked_ids, member_set):
        """Compute the running enrichment score."""
        hits = np.array([1 if rid in member_set else 0 for rid in ranked_ids])
        miss = 1 - hits
        weights = np.array([abs(scores.get(rid, 0)) for rid in ranked_ids])
        hit_sum = (hits * weights).sum()
        if hit_sum == 0:
            return 0.0
        miss_count = miss.sum()
        if miss_count == 0:
            return 0.0
        running = np.cumsum(hits * weights / hit_sum - miss / miss_count)
        # ES = max deviation from zero
        pos_max = running.max()
        neg_min = running.min()
        return pos_max if abs(pos_max) >= abs(neg_min) else neg_min

    results = {}
    for pw_name, rxn_ids in pathway_definitions.items():
        member_set = set(rxn_ids) & set(all_ids)
        if len(member_set) < 2:
            continue
        observed_es = _running_es(sorted_ids, member_set)

        # Permutation test
        null_es = []
        for _ in range(n_permutations):
            perm = rng.permutation(sorted_ids)
            null_es.append(_running_es(perm, member_set))
        null_es = np.array(null_es)

        if observed_es >= 0:
            p = (null_es >= observed_es).mean()
            null_mean = null_es[null_es >= 0].mean() if (null_es >= 0).any() else 1.0
        else:
            p = (null_es <= observed_es).mean()
            null_mean = abs(null_es[null_es < 0].mean()) if (null_es < 0).any() else 1.0

        nes = observed_es / null_mean if null_mean != 0 else 0.0
        results[pw_name] = {"ES": observed_es, "NES": nes, "pval": max(p, 1.0 / n_permutations)}

    df = pd.DataFrame(results).T
    if not df.empty:
        df["padj"] = bh_adjust(df["pval"].values)
    return df


def over_representation_analysis(
    sig_ids: List[str],
    pathway_definitions: Dict[str, List[str]],
    background_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Over-Representation Analysis (ORA) with configurable background.

    Parameters
    ----------
    sig_ids : list of str
        IDs of significant reactions/genes.
    pathway_definitions : dict
        Mapping of pathway names to lists of IDs.
    background_ids : list of str, optional
        Background universe of IDs.  If ``None``, the union of all
        pathway members plus *sig_ids* is used.

    Returns
    -------
    pandas.DataFrame
        With columns ``pval``, ``padj``, ``BgRatio``, ``SigRatio``.
    """
    sig_set = set(sig_ids)
    if background_ids is None:
        all_members = set()
        for members in pathway_definitions.values():
            all_members.update(members)
        all_members.update(sig_set)
        background_ids = list(all_members)
    bg_set = set(background_ids)
    n_bg = len(bg_set)
    n_sig = len(sig_set & bg_set)

    p_vals, bg_ratio, sig_ratio = {}, {}, {}
    for pw_name, members in pathway_definitions.items():
        pw_in_bg = set(members) & bg_set
        n_pw = len(pw_in_bg)
        k = len(pw_in_bg & sig_set)
        if n_pw == 0:
            continue
        rv = hypergeom(n_bg, n_sig, n_pw)
        p_vals[pw_name] = rv.sf(k - 1) if k > 0 else 1.0
        bg_ratio[pw_name] = f"{n_pw}/{n_bg}"
        sig_ratio[pw_name] = f"{k}/{n_sig}"

    p_series = pd.Series(p_vals)
    return pd.DataFrame({
        "pval": p_series,
        "padj": pd.Series(bh_adjust(p_series), index=p_series.index),
        "BgRatio": bg_ratio,
        "SigRatio": sig_ratio,
    })