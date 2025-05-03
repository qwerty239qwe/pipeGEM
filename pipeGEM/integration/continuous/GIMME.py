from typing import Dict

from optlang.symbolics import Zero
import cobra
from cobra.util import fix_objective_as_constraint
import numpy as np
import pandas as pd

from pipeGEM.analysis import add_mod_pfba, GIMMEAnalysis, timing


@timing
def apply_GIMME(model: cobra.Model,
                rxn_expr_score: Dict[str, float],
                high_exp: float,
                protected_rxns = None,
                obj_frac: float = 0.8,
                remove_zero_fluxes: bool = False,
                flux_threshold: float = 1e-6,
                max_inconsistency_score = 1e3,
                return_fluxes: bool = True,
                keep_context: bool = False,
                rxn_scaling_coefs: dict = None,
                predefined_threshold=None
                ):
    """Apply the GIMME algorithm to generate a context-specific metabolic model.

    GIMME (Gene Inactivity Moderated by Metabolism and Expression) assumes that
    cellular metabolism aims to achieve a required metabolic functionality
    (defined by the model's objective function) with minimal deviation from a
    reference expression state. It minimizes the flux through reactions with
    expression below a threshold, subject to maintaining a certain level of
    the original objective function.

    Parameters
    ----------
    model : cobra.Model
        The input genome-scale metabolic model with a defined objective function
        representing the required metabolic functionality.
    rxn_expr_score : dict[str, float]
        Dictionary mapping reaction IDs to their expression scores. NaN values
        are ignored.
    high_exp : float
        Expression score threshold. Reactions with scores below this threshold
        are penalized in the GIMME objective function.
    protected_rxns : list[str], optional
        List of reaction IDs that should not be penalized, even if their
        expression is below `high_exp`. Defaults to None.
    obj_frac : float, optional
        Fraction of the original model's optimal objective value that must be
        maintained by the GIMME solution. Defaults to 0.8.
    remove_zero_fluxes : bool, optional
        If True, create a `result_model` by removing reactions with flux below
        `flux_threshold` in the GIMME solution. Defaults to False.
    flux_threshold : float, optional
        Flux threshold used when `remove_zero_fluxes` is True. Defaults to 1e-6.
    max_inconsistency_score : float, optional
        Value to cap the penalty applied to low-expression reactions to handle
        potential numerical issues with very low scores. Defaults to 1e3.
    return_fluxes : bool, optional
        If True, include the GIMME flux distribution in the result object.
        Defaults to True.
    keep_context : bool, optional
        If True, modify the input `model` by adding the GIMME objective and
        constraining the original objective. If False (default), modifications
        happen within a context manager.
    rxn_scaling_coefs : dict[str, float], optional
        Dictionary mapping reaction IDs to scaling coefficients, used to adjust
        objective weights and the removal `flux_threshold`. Defaults to None
        (all coeffs 1).
    predefined_threshold : any, optional
        This parameter is currently ignored by GIMME. Defaults to None.

    Returns
    -------
    GIMMEAnalysis
        An object containing the results:
        - rxn_coefficients (dict): Dictionary of objective coefficients (penalties)
          applied to low-expression reactions.
        - rxn_scores (dict): The input reaction expression scores.
        - flux_result (pd.DataFrame or None): GIMME flux distribution if
          `return_fluxes` is True.
        - result_model (cobra.Model or None): Pruned model if `remove_zero_fluxes`
          is True, otherwise None.

    Notes
    -----
    Based on the algorithm described in: Becker, S. A., & Palsson, B. Ø. (2008).
    Context-specific metabolic networks are consistent with experiments.
    PLoS computational biology, 4(5), e1000082.
    The objective function minimizes the sum of fluxes weighted by (high_exp - score)
    for reactions with score < high_exp.
    """
    protected_rxns = [] if protected_rxns is None else protected_rxns
    ori_obj = [r.id for r in model.reactions if r.objective_coefficient != 0]
    rxn_scaling_coefs = {r.id: 1 for r in model.reactions} if rxn_scaling_coefs is None else rxn_scaling_coefs
    obj_dict = {r_id: (high_exp - r_exp) * rxn_scaling_coefs[r_id]
                if high_exp - r_exp < max_inconsistency_score else max_inconsistency_score  # for preventing -np.inf values
                for r_id, r_exp in rxn_expr_score.items() if not np.isnan(r_exp) and
                r_id not in protected_rxns and r_id not in ori_obj and
                r_exp < high_exp}

    with model:
        add_mod_pfba(model, weights=obj_dict, fraction_of_optimum=obj_frac)
        sol = model.optimize("minimize")

    flux_df = sol.to_frame()
    print("original obj's optimized value: ", flux_df.loc[ori_obj, "fluxes"])

    new_model = None
    if remove_zero_fluxes:
        new_model = model.copy()
        to_remove = set(flux_df[abs(flux_df["fluxes"]).sort_index() <=
                                flux_threshold / pd.Series(rxn_scaling_coefs).sort_index()].index.to_list()) - set(protected_rxns)
        new_model.remove_reactions(list(to_remove), remove_orphans=True)

    if keep_context:
        rxns_in_model = [r.id for r in model.reactions]
        add_mod_pfba(model, weights={k: v for k, v in obj_dict.items() if k in rxns_in_model},
                     fraction_of_optimum=obj_frac) # some are probably removed

    result = GIMMEAnalysis(log={"name": model.name, "high_exp": high_exp, "obj_frac": obj_frac})
    result.add_result(dict(rxn_coefficents=obj_dict,
                           rxn_scores=rxn_expr_score,
                           flux_result=flux_df if return_fluxes else None,
                           result_model=new_model))
    return result
