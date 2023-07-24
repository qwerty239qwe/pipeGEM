import pandas as pd
from scipy.stats import hypergeom
from pipeGEM.analysis._utils import bh_adjust


class HyperGeometricTester:
    def __init__(self):
        pass


def hypergeometric_test(data: pd.DataFrame,
                        pathway_col: str,
                        sig_col: str) -> pd.DataFrame:
    """
    The function uses the hypergeometric test to determine if certain pathways have a significantly
    higher number of significant reactions compared to what would be expected by chance.

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