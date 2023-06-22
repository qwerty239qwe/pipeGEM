import pandas as pd
from scipy.stats import hypergeom
from pipeGEM.analysis._utils import bh_adjust


class HyperGeometricTester:
    def __init__(self):
        pass


def hypergeometric_test(data: pd.DataFrame,
                        pathway_col,
                        sig_col):
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