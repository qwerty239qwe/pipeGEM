from typing import Dict, List, Callable, Union

import cobra
from cobra.flux_analysis.parsimonious import pfba
import numpy as np

from pipeGEM.utils import select_rxns_from_model
from pipeGEM.utils.transform import exp_x, functions
from pipeGEM.analysis import EFluxAnalysis, timing


@timing
def apply_EFlux(model: cobra.Model,
                rxn_expr_score: Dict[str, float],
                max_ub: float = 1000,
                min_lb: float = 1e-6,
                min_score: float = -1e3,
                protected_rxns: Union[str, List[str], None] = None,
                flux_threshold: float = 1e-6,
                remove_zero_fluxes: bool = False,
                return_fluxes: bool = True,
                transform: Union[Callable, str] = exp_x) -> EFluxAnalysis:
    """
    Applies the EFlux algorithm to a metabolic model, using gene expression data
    to constrain reaction fluxes. The EFlux method scales gene expression values
    to the range of reaction bounds, effectively enforcing that highly expressed
    reactions have high fluxes, and vice versa. Returns an EFluxAnalysis object
    with the calculated reaction bounds, gene expression scores, and fluxes.

    Parameters
    ----------
    model : cobra.Model
        A COBRApy metabolic model to be analyzed.
    rxn_expr_score : dict
        A dictionary of reaction IDs (str) to gene expression scores (float).
    max_ub : float, optional
        The maximum upper bound value to apply to the most highly expressed reaction
        (default 1000).
    min_lb : float, optional
        The minimum lower bound value to apply to the least expressed reaction
        (default 1e-6).
    min_score : float, optional
        The minimum gene expression score to consider (default -1e3).
    protected_rxns : str, list of str, or None, optional
        A single reaction ID, list of reaction IDs, or None. Any reactions in this
        list will be excluded from the analysis (default None).
    return_fluxes : bool, optional
        Whether to return the resulting fluxes as a pandas DataFrame (default True).
    transform : callable, optional
        A user-specified transformation function to apply to gene expression scores
        before scaling. Must take a single float argument and return a float.

    Returns
    -------
    EFluxAnalysis
        An EFluxAnalysis object containing the calculated reaction bounds, gene
        expression scores, and optionally, the resulting fluxes.
    """
    assert max_ub > 0, "max_ub should be a positive number"
    assert min_lb >= 0, "min_lb should be zero or a positive number"
    assert max_ub - min_lb > 0, "max_ub should be larger than min_lb"
    if protected_rxns:
        ignore_rxn_ids = select_rxns_from_model(model, protected_rxns, return_id=True)
        print(f"Ignoring {len(ignore_rxn_ids)} reactions ({ignore_rxn_ids[:10]}"
              f"{'...' if len(ignore_rxn_ids) > 10 else ''}) "
              f", no constraints will be applied on them")
    else:
        ignore_rxn_ids = []
    exps = [v if v > min_score else min_score for _, v in rxn_expr_score.items() if not np.isnan(v)]
    max_exp = max(exps) if len(exps) != 0 else max_ub
    min_exp = min(exps) if len(exps) != 0 else min_lb
    print(f"Max expression: {max_exp} | Min expression: {min_exp}")
    assert max_exp > 0, "max_exp should be a positive number, all expression values might be zeros"
    if isinstance(transform, str):
        if transform in functions:
            transform = functions[transform]
        else:
            transform = getattr(np, transform)

    trans_min_exp = transform(min_exp)
    denominator = transform(max_exp) - trans_min_exp
    if not np.isfinite(denominator):
        raise ValueError("Infinite or NaN denominator")
    capped_scores = {k: v if v > min_score else min_score if not np.isnan(v) else v for k, v in rxn_expr_score.items()}
    trans_rxn_exp_dict = {k: min_lb + ((max_ub - min_lb) * (transform(v) - trans_min_exp) / denominator)
                          if not np.isnan(v) else v
                          for k, v in capped_scores.items()}
    print(denominator, trans_min_exp)
    assert all([v >= 0 for v in trans_rxn_exp_dict.values() if not np.isnan(v)]), trans_rxn_exp_dict
    r_bounds_dict = {}
    model = model.copy()
    for r in model.reactions:
        if r not in model.exchanges and r.id not in ignore_rxn_ids:
            if not np.isnan(trans_rxn_exp_dict[r.id]) and (-trans_rxn_exp_dict[r.id] > r.lower_bound):
                r.lower_bound = -trans_rxn_exp_dict[r.id]
            if not np.isnan(trans_rxn_exp_dict[r.id]) and (trans_rxn_exp_dict[r.id] < r.upper_bound):
                r.upper_bound = trans_rxn_exp_dict[r.id]
        r_bounds_dict[r.id] = r.bounds
    sol = pfba(model)
    flux_df = sol.to_frame()
    if remove_zero_fluxes:
        to_remove = set(flux_df[abs(flux_df["fluxes"]) <= flux_threshold].index.to_list()) - set(protected_rxns)
        model.remove_reactions(list(to_remove), remove_orphans=True)
    result = EFluxAnalysis(log={"name": model.name,
                                "max_ub": max_ub,
                                "min_lb": min_lb,
                                "protected_rxns": protected_rxns,
                                "remove_zero_fluxes": remove_zero_fluxes})
    result.add_result(dict(rxn_bounds=r_bounds_dict,
                           rxn_scores=rxn_expr_score,
                           flux_result=flux_df if return_fluxes else None,
                           result_model=model))
    return result
