from typing import Literal

import numpy as np
import pandas as pd

from pipeGEM.integration.algo.iMAT import add_iMAT_cons_to_model, get_ind_var_for_rxns
from pipeGEM.integration.utils import *
from pipeGEM.analysis import timing, INIT_Analysis


def calc_INIT_weight(rxn_scores,
                     exp_th=None,
                     non_exp_th=None,
                     method: Literal["default", "threshold"] = "default",
                     absent_weight: float = -5,
                     ):
    if method == "default":
        return {r: 5 * np.log(v) for r, v in rxn_scores.items()}
    elif method == "threshold":
        z = np.polyfit([non_exp_th, exp_th], [0, 20], 1)
        p = np.poly1d(z)
        return {r: max(absent_weight, p(v)) if np.isfinite(v) else absent_weight for r, v in rxn_scores.items()}


@timing
def apply_INIT(model,
               data,
               predefined_threshold,
               threshold_kws: dict,
               protected_rxns=None,
               eps=1e-6,
               tol=1e-6,
               weight_method: Literal["default", "threshold"] = "threshold",
               rxn_scaling_coefs: dict = None,) -> INIT_Analysis:
    """Apply the INIT algorithm to generate a context-specific metabolic model.

    INIT (Integrative Network Inference for Tissues) uses expression data to
    assign weights to reactions. It then solves a mixed-integer linear
    programming (MILP) problem, similar to iMAT, to find a flux distribution
    that maximizes the sum of weights for active reactions. Reactions with
    near-zero flux in the optimal solution are removed.

    Parameters
    ----------
    model : cobra.Model
        The input genome-scale metabolic model.
    data : object
        An object containing gene expression data (`data.gene_data`) and
        reaction scores (`data.rxn_scores`) derived from it.
    predefined_threshold : dict or analysis_types
        Strategy or dictionary defining thresholds (`exp_th`, `non_exp_th`) used
        for weight calculation if `weight_method` is 'threshold'. See
        `pipeGEM.integration.utils.parse_predefined_threshold`.
    threshold_kws : dict
        Additional keyword arguments for the thresholding function if
        `weight_method` is 'threshold'.
    protected_rxns : list[str], optional
        A list of reaction IDs that should always be treated as core reactions
        and potentially assigned a high weight. Defaults to None.
    eps : float, optional
        Small flux value used in constraints to enforce activity through core
        reactions selected by the MILP (inherited from iMAT constraints).
        Defaults to 1e-6.
    tol : float, optional
        Flux tolerance threshold. Reactions with absolute flux below this value
        in the MILP solution are removed from the final model. Defaults to 1e-6.
    weight_method : {'default', 'threshold'}, optional
        Method to calculate reaction weights from scores:
        - 'default': Uses 5 * log(score).
        - 'threshold': Uses linear interpolation based on `exp_th` and `non_exp_th`.
        Defaults to "threshold".
    rxn_scaling_coefs : dict[str, float], optional
        Dictionary mapping reaction IDs to scaling coefficients. Currently unused
        in the main logic but potentially used for tolerance adjustment.
        Defaults to None.

    Returns
    -------
    INIT_Analysis
        An object containing the results:
        - result_model (cobra.Model): The final context-specific model.
        - removed_rxn_ids (np.ndarray): IDs of removed reactions.
        - threshold_analysis (ThresholdAnalysis or None): Details of thresholding
          used if `weight_method` was 'threshold'.
        - weight_dic (dict): Dictionary of calculated weights used in the objective.
        - fluxes (pd.DataFrame): DataFrame of absolute fluxes from the MILP solution.

    Notes
    -----
    Based on the algorithm described in: Agren, R., Bordel, S., Mardinoglu, A.,
    Pornputtapong, N., Nookaew, I., & Nielsen, J. (2012). Reconstruction of
    genome-scale active metabolic networks for 69 human cell types and 16 cancer
    types using INIT. PLoS computational biology, 8(5), e1002518.
    The implementation leverages the MILP formulation structure from iMAT.
    """
    gene_data, rxn_scores = data.gene_data, data.rxn_scores
    if weight_method == "threshold":
        threshold_dic = parse_predefined_threshold(predefined_threshold,
                                                   gene_data=gene_data,
                                                   **threshold_kws)
        th_result, exp_th, non_exp_th = threshold_dic["th_result"], threshold_dic["exp_th"], threshold_dic["non_exp_th"]
    else:
        th_result = None
        non_exp_th = 0
        exp_th = 0

    weight_dic = calc_INIT_weight(rxn_scores=rxn_scores,
                                  exp_th=exp_th, non_exp_th=non_exp_th, method=weight_method)
    model = model.copy()
    result_model = model.copy()
    core_rxn_ids = [r.id for r in model.reactions if rxn_scores[r.id] >= non_exp_th]
    non_core_rxn_ids = [r.id for r in model.reactions if rxn_scores[r.id] < non_exp_th]

    core_rxn_ids = list(set(core_rxn_ids) | set(protected_rxns))
    for r in protected_rxns:
        weight_dic[r] = max(weight_dic[r], 20)

    non_core_rxn_ids = list(set(non_core_rxn_ids) - set(protected_rxns))

    core_rxn_lbs = {r: model.reactions.get_by_id(r).lower_bound for r in core_rxn_ids}
    core_rxn_ubs = {r: model.reactions.get_by_id(r).upper_bound for r in core_rxn_ids}

    non_core_rxn_lbs = {r: model.reactions.get_by_id(r).lower_bound for r in non_core_rxn_ids}
    non_core_rxn_ubs = {r: model.reactions.get_by_id(r).upper_bound for r in non_core_rxn_ids}

    model.objective.set_linear_coefficients({v: 0 for v in model.variables})
    new_objs = {}

    core_f_ind_vars, core_b_ind_vars, non_core_ind_vars = get_ind_var_for_rxns(model=model,
                                                                               core_rxn_ids=core_rxn_ids,
                                                                               non_core_rxn_ids=non_core_rxn_ids)

    for name, var in non_core_ind_vars.items():
        new_objs[var] = abs(weight_dic[name[4:]])
    for name, var in core_f_ind_vars.items():
        new_objs[var] = weight_dic[name[4:]]
    for name, var in core_b_ind_vars.items():
        new_objs[var] = weight_dic[name[4:]]

    add_iMAT_cons_to_model(model=model,
                           core_f_ind_vars=core_f_ind_vars,
                           core_b_ind_vars=core_b_ind_vars,
                           non_core_ind_vars=non_core_ind_vars,
                           core_rxn_ids=core_rxn_ids,
                           core_rxn_lbs=core_rxn_lbs,
                           core_rxn_ubs=core_rxn_ubs,
                           non_core_rxn_ids=non_core_rxn_ids,
                           non_core_rxn_lbs=non_core_rxn_lbs,
                           non_core_rxn_ubs=non_core_rxn_ubs,
                           eps=eps)
    model.objective.set_linear_coefficients(new_objs)
    model.solver.update()
    sol = model.optimize("maximize")

    fluxes = abs(sol.to_frame()["fluxes"])
    if rxn_scaling_coefs is not None:
        tol_ = pd.Series({rxn_scaling_coefs[r] * tol for r in fluxes.index}, index=fluxes.index)
    else:
        tol_ = tol

    removed_rxn_ids = fluxes[fluxes < tol_].index.to_list()
    result_model.remove_reactions(removed_rxn_ids, remove_orphans=True)

    result = INIT_Analysis(log=dict(threshold_kws=threshold_kws,
                                    protected_rxns=protected_rxns,
                                    eps=eps,
                                    tol=tol,
                                    weight_method=weight_method))

    result.add_result(dict(result_model=result_model,
                           removed_rxn_ids=np.array(removed_rxn_ids),
                           threshold_analysis=th_result if weight_method == "threshold" else None,
                           weight_dic=weight_dic,
                           fluxes=fluxes.to_frame()))

    return result
