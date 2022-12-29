from typing import Dict

import cobra

from cobra.util import fix_objective_as_constraint
import numpy as np
from pipeGEM.analysis import modified_pfba, add_mod_pfba, RIPTiDePruningAnalysis, RIPTiDeSamplingAnalysis, flux_analyzers, timing


@timing
def apply_RIPTiDe_pruning(model,
                          rxn_expr_score: Dict[str, float],
                          max_gw: float = None,
                          obj_frac: float = 0.8,
                          threshold: float = 1e-6,
                          protected_rxns = None,
                          max_inconsistency_score = 1e3,
                          **kwargs
                          ):
    if protected_rxns is None:
        protected_rxns = []
    rxn_expr_score = {k: v if -max_inconsistency_score < v < max_inconsistency_score else max_inconsistency_score
                      if v > max_inconsistency_score else -max_inconsistency_score
                      for k, v in rxn_expr_score.items() if not np.isnan(v)}

    max_gw = max_gw or max([i for i in rxn_expr_score.values() if not np.isnan(i)])
    if np.isnan(max_gw):
        raise ValueError("max_gw cannot be NaN")
    min_gw = min([i for i in rxn_expr_score.values() if np.isfinite(i)])
    print(f"Max RAL: {max_gw}, Min RAL: {min_gw}")
    obj_dict = {r_id: (max_gw - r_exp) / (max_gw - min_gw) if (max_gw + min_gw - r_exp) < max_inconsistency_score else 1
                for r_id, r_exp in rxn_expr_score.items() if not (np.isnan(r_exp) or r_id in protected_rxns)}

    if not all([0 <= v <= 1 for _, v in obj_dict.items()]):
        raise ValueError(f"Some of the obj values are invalid, {[v for _, v in obj_dict.items() if not (0 <= v <= 1)]}")
    sol_df = modified_pfba(model, weights=obj_dict, fraction_of_optimum=obj_frac).to_frame()
    rxn_to_remove = list(set(sol_df[abs(sol_df["fluxes"]) < threshold].index.to_list()) - set(protected_rxns))
    output_model = model.copy()
    output_model.remove_reactions(rxn_to_remove, remove_orphans=True)
    result = RIPTiDePruningAnalysis(log={"name": model.name, "max_gw": max_gw, "obj_frac": obj_frac,
                                         "threshold": threshold})
    result.add_result(model= output_model, removed_rxns=rxn_to_remove, obj_dict=obj_dict)
    return result


@timing
def apply_RIPTiDe_sampling(model,
                           rxn_expr_score: Dict[str, float],
                           max_gw: float = None,
                           max_inconsistency_score = 1e3,
                           obj_frac: float = 0.8,
                           sampling_obj_frac: float = 0.05,
                           do_sampling: bool = False,
                           solver = "gurobi",
                           sampling_method: str = "gapsplit",
                           protected_rxns = None,
                           sampling_n: int = 500,
                           keep_context: bool = False,
                           **kwargs
                           ):
    rxn_expr_score = {k: v if -max_inconsistency_score < v < max_inconsistency_score else max_inconsistency_score
                      if v > max_inconsistency_score else -max_inconsistency_score
                      for k, v in rxn_expr_score.items() if not np.isnan(v)}
    max_gw = max_gw or np.nanmax(list(rxn_expr_score.values()))
    min_gw = np.nanmin(list(rxn_expr_score.values()))
    print(f"Max RAL: {max_gw}, Min RAL: {min_gw}")
    protected_rxns = protected_rxns or []
    if max_gw < max(rxn_expr_score.values()):
        raise ValueError("max_gw must be greater than or equal to the max rxn score")
    obj_dict = {r_id: (r_exp - min_gw) / (max_gw - min_gw)
                for r_id, r_exp in rxn_expr_score.items() if not np.isnan(r_exp)}
    obj_dict.update({r.id: 1
                     for r in model.reactions if r.id not in obj_dict or r.id in protected_rxns})  # same as the smallest weight
    assert all([1 >= i >= 0 for i in list(obj_dict.values())])

    sampling_result = None
    if do_sampling:
        with model:
            add_mod_pfba(model, weights=obj_dict, fraction_of_optimum=obj_frac, direction="max")
            sol = model.optimize()
            print(sol.to_frame(), sol.objective_value)
            sampling_analyzer = flux_analyzers["sampling"](model, solver, log={"n": sampling_n,
                                                                               "method": sampling_method,
                                                                               **kwargs})
            sampling_result = sampling_analyzer.analyze(n=sampling_n,
                                                        method=sampling_method,
                                                        obj_lb_ratio=sampling_obj_frac,
                                                        **kwargs)
    if keep_context:
        add_mod_pfba(model, weights=obj_dict, fraction_of_optimum=obj_frac, direction="max")
        fix_objective_as_constraint(model=model, fraction=sampling_obj_frac)
    analysis_result = RIPTiDeSamplingAnalysis(log = {"max_gw": max_gw, "obj_frac": obj_frac,
                                                     "sampling_obj_frac": sampling_obj_frac,
                                                     "do_sampling": do_sampling, "solver": solver,
                                                     "sampling_method": sampling_method})
    analysis_result.add_result(sampling_result=sampling_result)
    return analysis_result
