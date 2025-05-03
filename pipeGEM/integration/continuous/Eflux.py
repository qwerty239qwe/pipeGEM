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
                transform: Union[Callable, str] = exp_x,
                rxn_scaling_coefs: Dict[str, float] = None,
                predefined_threshold=None) -> EFluxAnalysis:
    """Apply the E-Flux algorithm to constrain model fluxes based on expression.

    E-Flux uses reaction expression scores (e.g., derived from transcriptomics)
    to set reaction bounds. Scores are typically transformed and then linearly
    scaled to map the expression range [min_exp, max_exp] to the flux range
    [min_lb, max_ub]. This enforces higher flux capacity for highly expressed
    reactions. Parsimonious FBA (pFBA) is then run on the constrained model.

    Parameters
    ----------
    model : cobra.Model
        The input genome-scale metabolic model.
    rxn_expr_score : dict[str, float]
        Dictionary mapping reaction IDs to their expression scores. NaN values
        are handled. Scores below `min_score` are capped.
    max_ub : float, optional
        The maximum flux bound assigned to the reaction(s) with the highest
        (transformed) expression score. Defaults to 1000.
    min_lb : float, optional
        The minimum flux bound assigned to the reaction(s) with the lowest
        (transformed) expression score. Defaults to 1e-6.
    min_score : float, optional
        Minimum expression score to consider; scores below this are capped at
        this value before transformation and scaling. Defaults to -1e3.
    protected_rxns : str or list[str] or None, optional
        Reaction ID(s) to exclude from bound constraints. Defaults to None.
    flux_threshold : float, optional
        Flux threshold used when `remove_zero_fluxes` is True. Reactions with
        absolute pFBA flux below this are removed. Defaults to 1e-6.
    remove_zero_fluxes : bool, optional
        If True, remove reactions with pFBA flux below `flux_threshold` from
        the final model. Defaults to False.
    return_fluxes : bool, optional
        If True, include the pFBA flux distribution in the result object.
        Defaults to True.
    transform : callable or str, optional
        Function or name of a function (from `pipeGEM.utils.transform.functions`
        or `numpy`) to apply to expression scores before scaling (e.g., `exp_x`).
        Defaults to `exp_x`.
    rxn_scaling_coefs : dict[str, float], optional
        Dictionary mapping reaction IDs to scaling coefficients. Applied *after*
        scaling expression to bounds (divides the calculated bound).
        Defaults to None (all coeffs 1).
    predefined_threshold : any, optional
        This parameter is currently ignored by E-Flux. Defaults to None.

    Returns
    -------
    EFluxAnalysis
        An object containing the results:
        - rxn_bounds (dict): Dictionary of the final bounds applied to each reaction.
        - rxn_scores (dict): The input reaction expression scores.
        - flux_result (pd.DataFrame or None): pFBA flux distribution if `return_fluxes` is True.
        - result_model (cobra.Model): The model with E-Flux bounds applied (and
          potentially pruned if `remove_zero_fluxes` is True).

    Raises
    ------
    AssertionError
        If `max_ub` <= 0, `min_lb` < 0, `max_ub` <= `min_lb`, or `max_exp` <= 0.
    ValueError
        If the denominator used for scaling becomes non-finite (e.g., due to
        `transform` function behavior or `max_exp` == `min_exp`).

    Notes
    -----
    Based on the method described in: Colijn, C., Brandes, A., Zucker, J., Lun, D. S.,
    Wienecke, A., Romaszko, J., ... & Ekins, S. (2009). Interpreting expression data
    with metabolic flux models: predicting Mycobacterium tuberculosis mycolic acid
    production. PLoS computational biology, 5(8), e1000489. (Though the implementation
    details like transformation and scaling might differ).
    Exchange reactions are typically excluded from bound setting.
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

    if rxn_scaling_coefs is not None:
        print("Identified rxn_scaling_coefs, will use it to adjust flux values")
    rxn_scaling_coefs = {r.id: 1 for r in model.reactions} if rxn_scaling_coefs is None else rxn_scaling_coefs
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
            coef = rxn_scaling_coefs[r.id] if r.id in rxn_scaling_coefs else 1
            if not np.isnan(trans_rxn_exp_dict[r.id]) and (-trans_rxn_exp_dict[r.id] / coef > r.lower_bound):
                r.lower_bound = -trans_rxn_exp_dict[r.id] / coef
            if not np.isnan(trans_rxn_exp_dict[r.id]) and (trans_rxn_exp_dict[r.id] / coef < r.upper_bound):
                r.upper_bound = trans_rxn_exp_dict[r.id] / coef
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
