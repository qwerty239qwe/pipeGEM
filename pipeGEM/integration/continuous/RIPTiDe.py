from typing import Dict, Optional, List

import cobra
from cobra.util import fix_objective_as_constraint
import numpy as np
import pandas as pd

from pipeGEM.analysis import modified_pfba, add_mod_pfba, RIPTiDePruningAnalysis, RIPTiDeSamplingAnalysis, flux_analyzers, timing


@timing
def apply_RIPTiDe_pruning(model,
                          rxn_expr_score: Dict[str, float],
                          max_gw: float = None,
                          obj_frac: float = 0.8,
                          threshold: float = 1e-6,
                          protected_rxns = None,
                          max_inconsistency_score = 1e3,
                          rxn_scaling_coefs: Dict[str, float] = None,
                          **kwargs
                          ):
    """Apply the pruning step of the RIPTiDe algorithm.

    This step uses parsimonious Flux Balance Analysis (pFBA) with weights
    derived from reaction expression scores (or RALs - Reaction Activity Levels)
    to identify and remove low-flux reactions, creating a pruned,
    context-specific model.

    Parameters
    ----------
    model : cobra.Model
        The input genome-scale metabolic model.
    rxn_expr_score : dict[str, float]
        Dictionary mapping reaction IDs to their expression scores (RALs).
        NaN values are ignored. Scores outside [-max_inconsistency_score,
        max_inconsistency_score] are capped.
    max_gw : float, optional
        Maximum possible reaction expression score (RAL). If None, it's
        calculated as the maximum finite value in `rxn_expr_score`.
        Defaults to None.
    obj_frac : float, optional
        Fraction of the optimal objective value to maintain when minimizing
        fluxes during pFBA. Defaults to 0.8.
    threshold : float, optional
        Flux threshold below which reactions are considered inactive and
        removed. Adjusted by `rxn_scaling_coefs` if provided. Defaults to 1e-6.
    protected_rxns : list[str], optional
        List of reaction IDs that should not be removed, even if their flux
        is below the threshold. Defaults to None.
    max_inconsistency_score : float, optional
        Value to cap reaction scores at (positive and negative) to handle
        extreme outliers. Defaults to 1e3.
    rxn_scaling_coefs : dict[str, float], optional
        Dictionary mapping reaction IDs to scaling coefficients. Used to adjust
        pFBA weights and the removal `threshold`. Defaults to None (all coeffs 1).
    **kwargs
        Additional keyword arguments (currently unused).

    Returns
    -------
    RIPTiDePruningAnalysis
        An object containing the results:
        - result_model (cobra.Model): The pruned context-specific model.
        - removed_rxn_ids (list[str]): List of IDs of removed reactions.
        - obj_dict (dict[str, float]): Dictionary of weights used in pFBA.

    Raises
    ------
    ValueError
        If `max_gw` is NaN after calculation or if derived pFBA objective
        coefficients are outside the expected [0, 1] range (after scaling).

    Notes
    -----
    RIPTiDe (Reaction Inclusion by Parsimony and Transcript Distribution) aims
    to create context-specific models reflecting metabolic activity based on
    transcriptomic data. This pruning step is the first part.
    Original paper: Jenior, M. L., et al. (2021). Transcriptome-guided parsimonious flux
    analysis improves predictions with metabolic networks in complex environments. 
    PLoS computational biology, 16(4), e1007099.
    """
    if protected_rxns is None:
        protected_rxns = []
    rxn_expr_score = {k: v if -max_inconsistency_score < v < max_inconsistency_score else max_inconsistency_score
                      if v > max_inconsistency_score else -max_inconsistency_score
                      for k, v in rxn_expr_score.items() if not np.isnan(v)}
    rxn_scaling_coefs = {r.id: 1 for r in model.reactions} if rxn_scaling_coefs is None else rxn_scaling_coefs
    max_gw = max_gw or max([i for i in rxn_expr_score.values() if not np.isnan(i)])
    if np.isnan(max_gw):
        raise ValueError("max_gw cannot be NaN")
    min_gw = min([i for i in rxn_expr_score.values() if np.isfinite(i)])
    print(f"Max RAL: {max_gw}, Min RAL: {min_gw}")
    obj_dict = {r_id: (max_gw - r_exp) * rxn_scaling_coefs[r_id] / (max_gw - min_gw)
                if (max_gw + min_gw - r_exp) < max_inconsistency_score else rxn_scaling_coefs[r_id]
                for r_id, r_exp in rxn_expr_score.items() if not (np.isnan(r_exp) or r_id in protected_rxns)}

    if not all([0 <= v <= 1 for _, v in obj_dict.items()]):
        raise ValueError(f"Some of the obj values are invalid, {[v for _, v in obj_dict.items() if not (0 <= v <= 1)]}")
    sol_df = modified_pfba(model, weights=obj_dict, fraction_of_optimum=obj_frac).to_frame()
    rxn_to_remove = list(set(sol_df[abs(sol_df["fluxes"]).sort_index() <
                                    threshold / pd.Series(rxn_scaling_coefs).sort_index()].index.to_list()) -
                         set(protected_rxns))
    output_model = model.copy()
    output_model.remove_reactions(rxn_to_remove, remove_orphans=True)
    result = RIPTiDePruningAnalysis(log={"name": model.name, "max_gw": max_gw, "obj_frac": obj_frac,
                                         "threshold": threshold})
    result.add_result(dict(result_model=output_model,
                           removed_rxn_ids=rxn_to_remove,
                           obj_dict=obj_dict))
    return result


@timing
def apply_RIPTiDe_sampling(model,
                           rxn_expr_score: Dict[str, float],
                           max_gw: float = None,
                           max_inconsistency_score: float = 1e3,
                           obj_frac: float = 0.8,
                           sampling_obj_frac: float = 0.8,
                           do_sampling: bool = False,
                           solver: str = "gurobi",
                           sampling_method: str = "gapsplit",
                           protected_rxns: Optional[List[str]] = None,
                           protect_no_expr: bool = False,
                           sampling_n: int = 500,
                           keep_context: bool = False,
                           rxn_scaling_coefs: Dict[str, float] = None,
                           discard_inf_score=True,
                           thinning=1,
                           processes=1,
                           seed=None,
                           **kwargs
                           ):
    """Apply the sampling step of the RIPTiDe algorithm or prepare for it.

    This step uses reaction expression scores (RALs) to define an objective
    function maximizing flux through high-expression reactions. It can optionally
    perform flux sampling on the model constrained by this objective.

    Parameters
    ----------
    model : cobra.Model
        The input metabolic model, typically the result of RIPTiDe pruning.
    rxn_expr_score : dict[str, float]
        Dictionary mapping reaction IDs to their expression scores (RALs).
        NaN values are ignored. Scores outside [-max_inconsistency_score,
        max_inconsistency_score] are capped unless `discard_inf_score` is True.
    max_gw : float, optional
        Maximum possible reaction expression score (RAL). If None, it's
        calculated as the maximum finite value in `rxn_expr_score`.
        Defaults to None.
    max_inconsistency_score : float, optional
        Value to cap reaction scores at (positive and negative) if
        `discard_inf_score` is False. Defaults to 1e3.
    obj_frac : float, optional
        Fraction of the optimal objective value (based on maximizing flux
        through high-RAL reactions) to use as a constraint if `keep_context`
        is True or during sampling setup. Defaults to 0.8.
    sampling_obj_frac : float, optional
        Fraction of the optimal objective value to maintain during flux
        sampling (passed to the sampler). Defaults to 0.8.
    do_sampling : bool, optional
        If True, perform flux sampling after setting up the objective and
        constraints. If False, only sets up the model context. Defaults to False.
    solver : str, optional
        Solver to use for optimization and sampling (e.g., 'gurobi', 'cplex').
        Defaults to "gurobi".
    sampling_method : str, optional
        Flux sampling algorithm to use ('achr', 'optgp', 'gapsplit').
        Defaults to "gapsplit".
    protected_rxns : list[str], optional
        List of reaction IDs to assign the maximum weight in the objective,
        regardless of their RAL. Defaults to None.
    protect_no_expr : bool, optional
        If True, assign maximum weight to reactions not present in
        `rxn_expr_score`. Defaults to False.
    sampling_n : int, optional
        Number of flux samples to generate if `do_sampling` is True.
        Defaults to 500.
    keep_context : bool, optional
        If True, modify the input `model` by adding the RIPTiDe objective
        and constraining it based on `obj_frac`. If False, modifications
        happen within a context manager only during sampling. Defaults to False.
    rxn_scaling_coefs : dict[str, float], optional
        Dictionary mapping reaction IDs to scaling coefficients, used to adjust
        objective weights. Defaults to None (all coeffs 1).
    discard_inf_score : bool, optional
        If True, treat infinite scores in `rxn_expr_score` as NaN (ignored).
        If False, cap them using `max_inconsistency_score`. Defaults to True.
    thinning : int, optional
        Thinning factor for flux sampling (passed to sampler). Defaults to 1.
    processes : int, optional
        Number of parallel processes for flux sampling. Defaults to 1.
    seed : int, optional
        Random seed for flux sampling. Defaults to None.
    **kwargs
        Additional keyword arguments passed to the flux sampler.

    Returns
    -------
    RIPTiDeSamplingAnalysis
        An object containing the results:
        - sampling_result (SamplingAnalysis or None): Results from flux sampling
          if `do_sampling` was True, otherwise None.

    Raises
    ------
    ValueError
        If `max_gw` is less than the maximum score in `rxn_expr_score`.

    Notes
    -----
    This function sets up the model for RIPTiDe-based flux analysis or sampling.
    The objective function maximizes flux weighted by scaled RALs.
    See: Jenior, M. L., et al. (2021). Transcriptome-guided parsimonious flux 
    analysis improves predictions with metabolic networks in complex environments. 
    PLoS computational biology, 16(4), e1007099.
    """
    if discard_inf_score:
        rxn_expr_score = {k: v if np.isfinite(v) else np.nan for k, v in rxn_expr_score.items()}

    rxn_expr_score = {k: v if -max_inconsistency_score <= v <= max_inconsistency_score else max_inconsistency_score
                      if v > max_inconsistency_score else -max_inconsistency_score
                      for k, v in rxn_expr_score.items() if not np.isnan(v)}
    max_gw = max_gw or np.nanmax(list(rxn_expr_score.values()))
    min_gw = np.nanmin(list(rxn_expr_score.values()))
    rxn_scaling_coefs = {r.id: 1 for r in model.reactions} if rxn_scaling_coefs is None else rxn_scaling_coefs
    print(f"Max RAL: {max_gw}, Min RAL: {min_gw}")
    protected_rxns = protected_rxns or []
    if max_gw < max(rxn_expr_score.values()):
        raise ValueError("max_gw must be greater than or equal to the max rxn score")
    obj_dict = {r_id: (r_exp - min_gw) * rxn_scaling_coefs[r_id] / (max_gw - min_gw)
                for r_id, r_exp in rxn_expr_score.items() if not np.isnan(r_exp)}
    obj_dict.update({r.id: rxn_scaling_coefs[r.id]
                     for r in model.reactions
                     if r.id in protected_rxns or (protect_no_expr and (r.id not in obj_dict))})  # same as the largest weight
    # assert all([1 >= i >= 0 for i in list(obj_dict.values())])
    sampling_result = None
    if do_sampling:
        with model:
            add_mod_pfba(model, weights=obj_dict, fraction_of_optimum=obj_frac, direction="max")
            sol = model.optimize()
            print(sol.to_frame())
            print("pFBA obj", sol.objective_value)
            sampling_analyzer = flux_analyzers["sampling"](model, solver, log={"n": sampling_n,
                                                                               "method": sampling_method,
                                                                               **kwargs})
            sampling_result = sampling_analyzer.analyze(n=sampling_n,
                                                        method=sampling_method,
                                                        obj_lb_ratio=sampling_obj_frac,
                                                        thinning=thinning,
                                                        processes=processes,
                                                        seed=seed)
    if keep_context:
        add_mod_pfba(model, weights=obj_dict, fraction_of_optimum=obj_frac, direction="max")
        fix_objective_as_constraint(model=model, fraction=sampling_obj_frac)
    analysis_result = RIPTiDeSamplingAnalysis(log = {"max_gw": max_gw,
                                                     "obj_frac": obj_frac,
                                                     "sampling_obj_frac": sampling_obj_frac,
                                                     "do_sampling": do_sampling, "solver": solver,
                                                     "sampling_method": sampling_method})
    analysis_result.add_result(dict(sampling_result=sampling_result))
    return analysis_result
