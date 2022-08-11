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
                          **kwargs
                          ):
    if protected_rxns is None:
        protected_rxns = []
    max_gw = max_gw or max([i for i in rxn_expr_score.values() if not np.isnan(i)])
    if max_gw < max(rxn_expr_score.values()):
        raise ValueError("max_gw must be greater than or equal to the max rxn score")
    if max_gw == 0:
        raise ValueError("max_gw cannot be zero")
    if np.isnan(max_gw):
        raise ValueError("max_gw cannot be NaN")
    min_gw = min([i for i in rxn_expr_score.values() if not np.isnan(i)])
    obj_dict = {r_id: (max_gw + min_gw - r_exp) / max_gw
                for r_id, r_exp in rxn_expr_score.items() if not np.isnan(r_exp)}
    obj_dict.update({r.id: min_gw / max_gw
                     for r in model.reactions if r.id not in obj_dict})  # same as the smallest weight
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
                           obj_frac: float = 0.8,
                           do_sampling: bool = False,
                           solver = "gurobi",
                           sampling_method: str = "gapsplit",
                           sampling_n: int = 500,
                           **kwargs
                           ):
    max_gw = max_gw or max(rxn_expr_score.values())
    if max_gw < max(rxn_expr_score.values()):
        raise ValueError("max_gw must be greater than or equal to the max rxn score")
    obj_dict = {r_id: r_exp / max_gw
                for r_id, r_exp in rxn_expr_score.items() if not np.isnan(r_exp)}
    obj_dict.update({r.id: 1
                     for r in model.reactions if r.id not in obj_dict})  # same as the smallest weight
    add_mod_pfba(model, weights=obj_dict, fraction_of_optimum=obj_frac, direction="max")
    fix_objective_as_constraint(model=model, fraction=obj_frac)
    sampling_result = None
    if do_sampling:
        sampling_analyzer = flux_analyzers["sampling"](model, solver, log={"n": sampling_n,
                                                                           "method": sampling_method,
                                                                           **kwargs})
        sampling_result = sampling_analyzer.analyze({"n": sampling_n,
                                                     "method": sampling_method,
                                                     **kwargs})
    analysis_result = RIPTiDeSamplingAnalysis(log = {"max_gw": max_gw, "obj_frac": obj_frac,
                                                     "do_sampling": do_sampling, "solver": solver,
                                                     "sampling_method": sampling_method})
    analysis_result.add_result(sampling_result=sampling_result)
    return analysis_result
