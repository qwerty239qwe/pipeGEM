from typing import Dict, List, Union, Optional

from optlang.symbolics import Zero
import cobra
from cobra.util import fix_objective_as_constraint
import numpy as np
import pandas as pd

from pipeGEM.analysis import add_mod_pfba, SPOTAnalysis, timing
from pipeGEM._logging import get_logger

logger = get_logger(__name__)


def _add_spot_norm_constraint(model: cobra.Model,
                              norm_ub: float = 1e4,
                              name: str = "spot_norm_constraint") -> None:
    """Add an L1 flux-sum constraint via the optlang API.

    This bounds the total absolute flux (sum of all forward and reverse
    variables) to *norm_ub*, preventing the optimizer from exploiting
    highly-expressed reactions at arbitrarily large flux values.
    Implemented purely through the optlang interface so it works with
    any COBRA-compatible solver (GLPK, CPLEX, Gurobi, …).

    Parameters
    ----------
    model : cobra.Model
        The model to add the constraint to (modified in place).
    norm_ub : float
        Upper bound on the sum of all reaction flux variables.
    name : str
        Name of the constraint (must be unique within the model).
    """
    norm_cons = model.problem.Constraint(Zero, name=name, lb=0, ub=norm_ub)
    model.add_cons_vars([norm_cons])
    model.solver.update()
    terms = {}
    for r in model.reactions:
        terms[r.forward_variable] = 1.0
        terms[r.reverse_variable] = 1.0
    norm_cons.set_linear_coefficients(terms)
    model.solver.update()


def _build_spot_objective(model: cobra.Model,
                          obj_weights: Dict[str, float],
                          obj_frac: float) -> None:
    """Set up the SPOT maximisation objective on *model* (in-place).

    Optionally fixes the existing FBA objective as a lower-bound
    constraint first (when *obj_frac* > 0), then replaces the model
    objective with a weighted-sum maximisation over *obj_weights*.

    Parameters
    ----------
    model : cobra.Model
        The model to modify.
    obj_weights : dict[str, float]
        Mapping of reaction ID → expression weight.  Only reactions
        present in the model are used.
    obj_frac : float
        Fraction of the FBA-optimal value to maintain as a constraint.
        Set to 0 to skip the FBA constraint entirely.
    """
    if obj_frac > 0:
        fix_objective_as_constraint(model, fraction=obj_frac)
    # fraction_of_optimum=0 so add_mod_pfba won't call
    # fix_objective_as_constraint a second time.
    add_mod_pfba(model, weights=obj_weights, fraction_of_optimum=0,
                 direction="max")


@timing
def apply_SPOT(model: cobra.Model,
               rxn_expr_score: Dict[str, float],
               protected_rxns: Optional[Union[str, List[str]]] = None,
               obj_frac: float = 0.1,
               norm_ub: float = 1e4,
               remove_zero_fluxes: bool = False,
               flux_threshold: float = 1e-6,
               return_fluxes: bool = True,
               keep_context: bool = False,
               rxn_scaling_coefs: Optional[Dict[str, float]] = None,
               predefined_threshold=None) -> SPOTAnalysis:
    """Apply the SPOT algorithm to generate an expression-guided flux distribution.

    SPOT (Simplified Phenotype Optimization Technique) finds a flux
    distribution that maximises the correlation between reaction fluxes and
    gene-expression scores while keeping the model's metabolic objective
    (e.g. biomass) at a user-specified fraction of its FBA-optimal value.

    The optimisation problem solved is:

    .. math::

        \\max \\sum_i w_i \\cdot v_i \\\\
        \\text{s.t.} \\quad f_{\\text{FBA}} \\geq \\texttt{obj\\_frac} \\cdot f^*_{\\text{FBA}} \\\\
        \\sum_i (v_i^+ + v_i^-) \\leq \\texttt{norm\\_ub} \\\\
        v \\in \\text{FBA feasible region}

    where :math:`w_i = \\texttt{rxn\\_expr\\_score}[i] \\cdot
    \\texttt{rxn\\_scaling\\_coefs}[i]`.

    Parameters
    ----------
    model : cobra.Model
        The input genome-scale metabolic model with a defined objective
        function representing the required metabolic functionality.
    rxn_expr_score : dict[str, float]
        Mapping of reaction IDs to expression scores.  ``NaN`` values are
        ignored.
    protected_rxns : str or list[str] or None, optional
        Reaction IDs excluded from the SPOT objective (their expression
        scores do not contribute to the weighted sum).  Defaults to None.
    obj_frac : float, optional
        Fraction of the FBA-optimal objective value that must be maintained
        as a lower-bound constraint during SPOT optimisation.  Set to 0 to
        omit the FBA constraint (free maximisation).  Defaults to 0.1.
    norm_ub : float, optional
        Upper bound for the L1 flux-sum constraint
        ``Σ(v_i^+ + v_i^-) ≤ norm_ub``.  Prevents the solver from
        exploiting highly-expressed reactions at unbounded flux.
        Defaults to 1e4.
    remove_zero_fluxes : bool, optional
        If ``True``, build a ``result_model`` by removing reactions whose
        absolute flux in the SPOT solution is ≤ ``flux_threshold``.
        Defaults to ``False``.
    flux_threshold : float, optional
        Flux cutoff used when ``remove_zero_fluxes=True``.
        Defaults to 1e-6.
    return_fluxes : bool, optional
        If ``True``, store the SPOT flux distribution in the result object.
        Defaults to ``True``.
    keep_context : bool, optional
        If ``True``, the SPOT modifications (FBA constraint, norm constraint,
        SPOT objective) are applied *permanently* to the input model.
        If ``False`` (default), all modifications are made inside a context
        manager and reverted afterwards.
    rxn_scaling_coefs : dict[str, float] or None, optional
        Per-reaction scaling coefficients that multiply the expression
        weights before forming the objective.  Defaults to None (all 1.0).
    predefined_threshold : any, optional
        Currently unused by SPOT; accepted for API consistency.
        Defaults to None.

    Returns
    -------
    SPOTAnalysis
        Object containing:

        - ``flux_result`` (*pandas.DataFrame* or None) — SPOT flux
          distribution if ``return_fluxes=True``.
        - ``result_model`` (*cobra.Model* or None) — pruned model if
          ``remove_zero_fluxes=True``, otherwise None.
        - ``rxn_scores`` (*dict*) — the original ``rxn_expr_score`` input.

    Notes
    -----
    Based on: Becker, S. A., & Palsson, B. Ø. (2008). Context-specific
    metabolic networks are consistent with experiments. *PLoS computational
    biology*, 4(5), e1000082.  (SPOT is a variant of this family of methods.)
    The L1 norm constraint is implemented directly via the optlang API so
    that the function works with GLPK, CPLEX, and Gurobi without requiring
    any solver-specific imports.
    """
    protected_rxns = [] if protected_rxns is None else (
        [protected_rxns] if isinstance(protected_rxns, str) else list(protected_rxns)
    )
    rxn_ids_in_model = {r.id for r in model.reactions}
    rxn_scaling_coefs = ({r.id: 1.0 for r in model.reactions}
                         if rxn_scaling_coefs is None else rxn_scaling_coefs)

    # Build expression weights keyed by reaction ID (strings).
    obj_weights = {
        r_id: r_exp * rxn_scaling_coefs.get(r_id, 1.0)
        for r_id, r_exp in rxn_expr_score.items()
        if not np.isnan(r_exp)
        and r_id not in protected_rxns
        and r_id in rxn_ids_in_model
    }

    def _apply_modifications(m: cobra.Model) -> cobra.Solution:
        _add_spot_norm_constraint(m, norm_ub=norm_ub)
        _build_spot_objective(m, obj_weights, obj_frac)
        sol = m.optimize("maximize")
        logger.info("SPOT objective value: %s", sol.objective_value)
        return sol

    if keep_context:
        sol = _apply_modifications(model)
    else:
        with model:
            sol = _apply_modifications(model)

    flux_df = sol.to_frame()

    new_model = None
    if remove_zero_fluxes:
        new_model = model.copy()
        to_remove = (
            set(flux_df[abs(flux_df["fluxes"]) <= flux_threshold].index.to_list())
            - set(protected_rxns)
        )
        new_model.remove_reactions(list(to_remove), remove_orphans=True)

    result = SPOTAnalysis(log={
        "name": model.name,
        "obj_frac": obj_frac,
        "norm_ub": norm_ub,
        "protected_rxns": protected_rxns,
        "remove_zero_fluxes": remove_zero_fluxes,
    })
    result.add_result(dict(
        rxn_scores=rxn_expr_score,
        flux_result=flux_df if return_fluxes else None,
        result_model=new_model,
    ))
    return result
