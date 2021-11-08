import copy
from tqdm import tqdm
from typing import Dict, List

import pandas as pd
import numpy as np
import cobra

from pipeGEM.utils import get_objective_rxn, make_irrev_rxn


def _corso_check_sol(model,
                     objective_sense,
                     constraint_option,
                     constraint_val,
                     obj_val,
                     sol_df,
                     obj_ids=None):
    if obj_ids is None:
        obj_ids = get_objective_rxn(model)
    if constraint_option == "percentage":
        constraint_val = abs(constraint_val)
        obj_vals = {obj_id: (sol_df.loc[obj_id, "fluxes"] if isinstance(sol_df, pd.DataFrame) else sol_df[obj_id]) *
                             constraint_val / 100 for obj_id in obj_ids}
    elif constraint_option == "value":
        if objective_sense == "maximum":
            assert obj_val > constraint_val, 'Objective Flux not attainable'
            obj_vals = {obj_id: constraint_val for obj_id in obj_ids}
        elif objective_sense == "minimum":
            assert obj_val < constraint_val, 'Objective Flux not attainable'
            obj_vals = {obj_id: constraint_val for obj_id in obj_ids}
        else:
            raise ValueError("objective_sense should be chosen from maximum and minimum")
    else:
        raise ValueError("Invalid constraint_option")
    return obj_vals


def corsoFBA(model,
             objective_sense: str = "maximum",
             constraint_val: float = 1,
             constraint_option: str = "percentage",
             rxn_cost_dict: Dict[str, float] = None,
             orig_rev_rxns: List[str] = None,
             penalty_met_id: str = None,
             penalty_rxn_id: str = None,
             noise_level: float = 1e-2) -> (float, pd.DataFrame):
    assert constraint_option in ["percentage", "value"]
    sol = model.optimize(objective_sense=objective_sense)
    sol_df = sol.to_frame()
    if sol.objective_value < 1e-6:
        return 0, pd.DataFrame(np.zeros(shape=(sol_df.shape[0], 1)), index=sol_df.index, columns=["fluxes"])

    obj_vals = _corso_check_sol(model, objective_sense, constraint_option, constraint_val, sol.objective_value, sol_df)

    with model:
        if penalty_met_id is None:
            pseudo_met = cobra.Metabolite(id="pseudo_met")
        else:
            pseudo_met = model.metabolites.get_by_id(penalty_met_id)

        if orig_rev_rxns is not None:
            rev_rxns = orig_rev_rxns
        else:
            rev_rxns = [r.id for r in model.reactions if r.reversibility is True]
            for r_id in rev_rxns:
                make_irrev_rxn(model, r_id)
                model.reactions.get_by_id(r_id).knock_out()

        noises = np.random.uniform(0, noise_level, size=(len(model.reactions)))
        for i, r in enumerate(model.reactions):
            r.add_metabolites({
                pseudo_met: noises[i] + (rxn_cost_dict[r.id] if r.id in rxn_cost_dict else rxn_cost_dict[r.id[3:]])
            })

        if penalty_rxn_id is None:
            pseudo_met_ex = cobra.Reaction(id="EX_pseudo_met", lower_bound=0, upper_bound=np.inf)
            pseudo_met_ex.add_metabolites({
                pseudo_met: -1
            })
            model.add_reactions([pseudo_met_ex])
        else:
            pseudo_met_ex = model.reactions.get_by_id(penalty_rxn_id)
        model.objective.set_linear_coefficients({pseudo_met_ex.forward_variable: 1.0,
                                                 pseudo_met_ex.reverse_variable: -1.0})  # added
        all_rxn_ids = [r.id for r in model.reactions]
        for rxn_id, obj_val in obj_vals.items():
            obj_r = model.reactions.get_by_id(rxn_id)
            obj_r.lower_bound, obj_r.upper_bound = obj_val, obj_val
            if f"_R_{obj_r.id}" in all_rxn_ids:
                model.reactions.get_by_id(f"_R_{obj_r.id}").knock_out()
            if f"_F_{obj_r.id}" in all_rxn_ids:
                model.reactions.get_by_id(f"_F_{obj_r.id}").knock_out()

        new_sol = model.optimize(objective_sense="minimum").to_frame()

        added_rxns = []
        for r_id in rev_rxns:
            if r_id not in obj_vals:
                new_sol.loc[r_id, "fluxes"] = new_sol.loc[f"_F_{r_id}", "fluxes"] - new_sol.loc[f"_R_{r_id}", "fluxes"]
                added_rxns.extend([f"_F_{r_id}", f"_R_{r_id}"])
        new_sol[new_sol["fluxes"] < 1e-8] = 0
    return sol.objective_value, new_sol.loc[[s for s in sol.to_frame().index.to_list()
                                             if s != penalty_rxn_id and s not in added_rxns], :]


def _pred_corsoFBA(model,
                   obj_ids,
                   sol_val,
                   sol_dic,
                   penalty_met_id,
                   penalty_rxn_id,
                   orig_rxns: List[str],
                   orig_rev_rxns: List[str] = None,
                   objective_sense: str = "maximum",
                   constraint_val: float = 1,
                   constraint_option: str = "percentage",
                   rxn_cost_dict: Dict[str, float] = None,
                   noise_level: float = 1e-2):
    """This is the version that takes predefined / precomputed arguments"""
    assert constraint_option in ["percentage", "value"]
    obj_vals = _corso_check_sol(model, objective_sense, constraint_option, constraint_val, sol_val, sol_dic, obj_ids)

    with model:
        pseudo_met = model.metabolites.get_by_id(penalty_met_id)
        noises = np.random.uniform(0, noise_level, size=(len(model.reactions)))
        for i, r in enumerate(model.reactions):
            if r.id == penalty_rxn_id:
                continue
            r.add_metabolites({
                pseudo_met: noises[i] + (rxn_cost_dict[r.id] if r.id in rxn_cost_dict else rxn_cost_dict[r.id[3:]])
            })

        pseudo_met_ex = model.reactions.get_by_id(penalty_rxn_id)
        model.objective.set_linear_coefficients({pseudo_met_ex.forward_variable: 1.0,
                                                 pseudo_met_ex.reverse_variable: -1.0})  # added
        all_rxn_ids = [r.id for r in model.reactions]
        for rxn_id, obj_val in obj_vals.items():
            obj_r = model.reactions.get_by_id(rxn_id)
            obj_r.lower_bound, obj_r.upper_bound = obj_val, obj_val
            if f"_R_{obj_r.id}" in all_rxn_ids:
                model.reactions.get_by_id(f"_R_{obj_r.id}").knock_out()
            if f"_F_{obj_r.id}" in all_rxn_ids:
                model.reactions.get_by_id(f"_F_{obj_r.id}").knock_out()

        new_sol = model.optimize(objective_sense="minimum").to_frame()

        added_rxns = []
        for r_id in orig_rev_rxns:
            if r_id not in obj_vals:
                new_sol.loc[r_id, "fluxes"] = new_sol.loc[f"_F_{r_id}", "fluxes"] - new_sol.loc[f"_R_{r_id}", "fluxes"]
                added_rxns.extend([f"_F_{r_id}", f"_R_{r_id}"])
        new_sol[new_sol["fluxes"] < 1e-8] = 0
    return new_sol.loc[orig_rxns, :]


def find_supp_rxns_by_corsoFBA(model,
                               sol_val,
                               sol_dic,
                               rxn_to_check: str,
                               rxn_sets: Dict[str, set],
                               rxn_to_rxn_set_dfs: Dict[str, pd.DataFrame],
                               rxn_cost_dict,
                               orig_rxns,
                               orig_rev_rxns,
                               penalty_met_id,
                               penalty_rxn_id,
                               n_times: int = 5,
                               constraint_val: float = 1.,
                               constraint_option: str = "percentage",
                               noise_level: float = 1e-2,
                               flux_thres: float = 1e-6,
                               direction="maximum") -> (bool, Dict[str, set]):
    assert direction in ["maximum", "minimum"]
    assert len(rxn_sets) == len(rxn_to_rxn_set_dfs)
    supp_rxns_dict = {set_name: set() for set_name in rxn_sets}
    for i in range(n_times):
        if i == 0 and sol_val < flux_thres:
            return True, supp_rxns_dict
        flux_df = _pred_corsoFBA(model,
                                 obj_ids=[rxn_to_check],
                                 sol_val=sol_val,
                                 sol_dic=sol_dic,
                                 penalty_met_id=penalty_met_id,
                                 penalty_rxn_id=penalty_rxn_id,
                                 orig_rxns=orig_rxns,
                                 orig_rev_rxns=orig_rev_rxns,
                                 objective_sense=direction,
                                 constraint_val=constraint_val,
                                 constraint_option=constraint_option,
                                 rxn_cost_dict=rxn_cost_dict,
                                 noise_level=noise_level)
        all_supp_rxns = set(flux_df[abs(flux_df["fluxes"]) > flux_thres].index)
        for set_name, rxn_set in rxn_sets.items():
            supp_rxns_dict[set_name] |= (rxn_set & all_supp_rxns)
            rxn_to_rxn_set_dfs[set_name].loc[rxn_to_check, supp_rxns_dict[set_name]] = 1
    return False, supp_rxns_dict


def get_rxn_cost_dict(sample: str,
                      HC_rxn_dic,
                      MC_rxn_dic,
                      NC_rxn_dic,
                      OT_rxn_dic,
                      HC_score,
                      MC_score,
                      NC_score,
                      OT_score):
    rxn_cost_dict = {r: HC_score for r in HC_rxn_dic[sample]}
    rxn_cost_dict.update({r: OT_score for r in OT_rxn_dic[sample]})
    rxn_cost_dict.update({r: MC_score for r in MC_rxn_dic[sample]})
    rxn_cost_dict.update({r: NC_score for r in NC_rxn_dic[sample]})
    return rxn_cost_dict


def get_relation_df(set_0, set_1) -> pd.DataFrame:
    return pd.DataFrame(np.zeros((len(set_0), len(set_1))), index=list(set_0), columns=list(set_1))


def find_HC_supp_rxns(sample_names,
                      ref_model,
                      penalty_met_id,
                      penalty_rxn_id,
                      HC_rxn_dic,
                      MC_rxn_dic,
                      NC_rxn_dic,
                      OT_rxn_dic,
                      HC_score,
                      MC_score,
                      NC_score,
                      OT_score):
    HC_to_MC_df_dic, HC_to_NC_df_dic = {}, {}
    modified_model = ref_model.copy()
    orig_rxns = [r.id for r in ref_model.reactions]
    rev_rxns = [r.id for r in modified_model.reactions if r.reversibility is True]
    penalty_met = cobra.Metabolite(id=penalty_met_id)
    penalty_rxn = cobra.Reaction(id=penalty_rxn_id)
    penalty_rxn.add_metabolites({
        penalty_met: -1
    })
    modified_model.add_reactions([penalty_rxn])
    for r_id in rev_rxns:
        make_irrev_rxn(modified_model, r_id)
        modified_model.reactions.get_by_id(r_id).knock_out()
    for r in modified_model.reactions:
        r.add_metabolites({penalty_met: 1e-7})

    for sample in sample_names:
        print("processing ", sample)
        blocked_HC_rxns = set()
        rxn_cost_dict = get_rxn_cost_dict(sample=sample,
                                          HC_rxn_dic=HC_rxn_dic,
                                          MC_rxn_dic=MC_rxn_dic,
                                          NC_rxn_dic=NC_rxn_dic,
                                          OT_rxn_dic=OT_rxn_dic,
                                          HC_score=HC_score,
                                          MC_score=MC_score,
                                          NC_score=NC_score,
                                          OT_score=OT_score)
        HC_to_MC_df = get_relation_df(HC_rxn_dic[sample], MC_rxn_dic[sample])
        HC_to_NC_df = get_relation_df(HC_rxn_dic[sample], NC_rxn_dic[sample])
        print("Number of HC rxns to check: ", len(HC_rxn_dic[sample]))
        with modified_model as model:
            assert (len(rxn_cost_dict) + len(rev_rxns) * 2 + 1) == len(model.reactions), f"{len(rxn_cost_dict) + len(rev_rxns) + 1} != {len(model.reactions)}"
            supp_rxns_dict = {"MC": set(), "NC": set()}
            for i, HC_rxn in enumerate(HC_rxn_dic[sample]):
                print(i, ", checking: ", HC_rxn)
                model.objective = HC_rxn
                rxn_sets = {"MC": MC_rxn_dic[sample], "NC": NC_rxn_dic[sample]}
                rxn_to_rxn_set_dfs = {"MC": HC_to_MC_df, "NC": HC_to_NC_df}
                is_blocked = True
                for dir in ["maximum", "minimum"]:
                    rxn = model.reactions.get_by_id(HC_rxn)
                    if (dir == "maximum" and rxn.upper_bound > 0) or (dir == "minimum" and rxn.lower_bound < 0):
                        obj_val = model.slim_optimize()
                        obj_dic = {HC_rxn: obj_val}
                        flag, supp_dict = find_supp_rxns_by_corsoFBA(model,
                                                                     sol_val=obj_val,
                                                                     sol_dic=obj_dic,
                                                                     rxn_to_check=HC_rxn,
                                                                     rxn_sets=rxn_sets,
                                                                     rxn_to_rxn_set_dfs=rxn_to_rxn_set_dfs,
                                                                     rxn_cost_dict=rxn_cost_dict,
                                                                     orig_rev_rxns=rev_rxns,
                                                                     orig_rxns=orig_rxns,
                                                                     penalty_met_id=penalty_met_id,
                                                                     penalty_rxn_id=penalty_rxn_id,
                                                                     direction=dir)
                        supp_rxns_dict["MC"] |= supp_dict["MC"]
                        supp_rxns_dict["NC"] |= supp_dict["NC"]
                        is_blocked = (is_blocked and flag)
                if is_blocked:
                    blocked_HC_rxns.add(HC_rxn)
        HC_rxn_dic[sample] -= blocked_HC_rxns
        HC_to_MC_df = HC_to_MC_df.loc[HC_rxn_dic[sample], :]
        HC_to_NC_df = HC_to_NC_df.loc[HC_rxn_dic[sample], :]
        for name, supp_rxns in supp_rxns_dict.items():
            HC_rxn_dic[sample] |= supp_rxns
            if name == "MC":
                MC_rxn_dic[sample] -= supp_rxns
            elif name == "NC":
                NC_rxn_dic[sample] -= supp_rxns
        HC_to_MC_df_dic[sample] = HC_to_MC_df
        HC_to_NC_df_dic[sample] = HC_to_NC_df

    return {"HC_to_MC_df_dic": HC_to_MC_df_dic,
            "HC_to_NC_df_dic": HC_to_NC_df_dic,
            "HC_rxn_dic": HC_rxn_dic,
            "MC_rxn_dic": MC_rxn_dic,
            "NC_rxn_dic": NC_rxn_dic,
            "modified_model": modified_model,
            "orig_rxns": orig_rxns,
            "rev_rxns": rev_rxns}


def find_MC_supp_rxns(sample_names,
                      modified_model,
                      rev_rxns,
                      orig_rxns,
                      penalty_met_id,
                      penalty_rxn_id,
                      HC_rxn_dic,
                      MC_rxn_dic,
                      NC_rxn_dic,
                      OT_rxn_dic,
                      HC_score,
                      MC_score,
                      NC_score,
                      OT_score,
                      NC_rescue_thres):
    blocked_MC_rxns_dic = {}
    MC_to_NC_df_dic = {}
    MC_rxn_dic = copy.deepcopy(MC_rxn_dic)
    NC_rxn_dic = copy.deepcopy(NC_rxn_dic)
    for sample in sample_names:
        blocked_MC_rxns = set()
        rxn_cost_dict = get_rxn_cost_dict(sample=sample,
                                          HC_rxn_dic=HC_rxn_dic,
                                          MC_rxn_dic=MC_rxn_dic,
                                          NC_rxn_dic=NC_rxn_dic,
                                          OT_rxn_dic=OT_rxn_dic,
                                          HC_score=HC_score,
                                          MC_score=MC_score,
                                          NC_score=NC_score,
                                          OT_score=OT_score)
        MC_to_NC_df = get_relation_df(MC_rxn_dic[sample], NC_rxn_dic[sample])
        with modified_model as model:
            for MC_rxn in MC_rxn_dic[sample]:
                model.objective = MC_rxn
                rxn_sets = {"NC": NC_rxn_dic[sample]}
                rxn_to_rxn_set_dfs = {"NC": MC_to_NC_df}
                is_blocked = True
                for dir in ["maximum", "minimum"]:
                    rxn = model.reactions.get_by_id(MC_rxn)
                    if (dir == "maximum" and rxn.upper_bound > 0) or (dir == "minimum" and rxn.lower_bound < 0):
                        obj_val = model.slim_optimize()
                        obj_dic = {MC_rxn: obj_val}
                        flag, supp_dict = find_supp_rxns_by_corsoFBA(model,
                                                                     rxn_to_check=MC_rxn,
                                                                     sol_val=obj_val,
                                                                     sol_dic=obj_dic,
                                                                     rxn_sets=rxn_sets,
                                                                     rxn_to_rxn_set_dfs=rxn_to_rxn_set_dfs,
                                                                     rxn_cost_dict=rxn_cost_dict,
                                                                     orig_rev_rxns=rev_rxns,
                                                                     orig_rxns=orig_rxns,
                                                                     penalty_met_id=penalty_met_id,
                                                                     penalty_rxn_id=penalty_rxn_id,
                                                                     direction=dir)
                        is_blocked = (is_blocked and flag)
                if is_blocked:
                    blocked_MC_rxns.add(MC_rxn)
        MC_rxn_dic[sample] -= blocked_MC_rxns
        MC_to_NC_df = MC_to_NC_df.loc[MC_rxn_dic[sample], :]
        NC_score = MC_to_NC_df.sum(axis=1)
        NC_score.index = MC_to_NC_df.columns
        rescued_NC = set(NC_score[NC_score > NC_rescue_thres].index)
        remained_NC = MC_to_NC_df.columns
        MC_rxn_dic[sample] |= rescued_NC
        NC_rxn_dic[sample] -= rescued_NC
        MC_to_NC_df = MC_to_NC_df.loc[:, rescued_NC]
        MC_to_NC_df = pd.concat([MC_to_NC_df,
                                 pd.DataFrame(np.zeros(len(rescued_NC),
                                                       len(remained_NC)),
                                              columns=remained_NC, index=list(rescued_NC))], axis=0)
        MC_to_NC_df_dic[sample] = MC_to_NC_df

    return {"MC_to_NC_df_dic": MC_to_NC_df_dic,
            "blocked_MC_rxns_dic": blocked_MC_rxns_dic,
            "MC_rxn_dic": MC_rxn_dic,
            "NC_rxn_dic": NC_rxn_dic}


def check_MC_feasibility(sample_names,
                         ref_model,
                         HC_rxn_dic,
                         MC_rxn_dic,
                         NC_rxn_dic,
                         MC_to_NC_df_dic,
                         flux_thres):
    blocked_MC_rxns = set()
    blocked_NCs_dic = {}
    MC_rxn_dic = copy.deepcopy(MC_rxn_dic)
    HC_rxn_dic = copy.deepcopy(HC_rxn_dic)
    for sample in sample_names:
        with ref_model as model:
            for r in NC_rxn_dic[sample]:
                model.reactions.get_by_id(r).knock_out()

            blocked_NCs_dic[sample] = set()
            for MC_rxn in MC_rxn_dic[sample]:
                MC_rxn_r = model.reactions.get_by_id(MC_rxn)
                model.objective = MC_rxn
                is_blocked = ((MC_rxn_r.lower_bound >= 0 or
                               abs(model.optimize(objective_sense="minimum").objective_value) < flux_thres) and
                              (MC_rxn_r.upper_bound <= 0 or
                               abs(model.optimize(objective_sense="maximum").objective_value) < flux_thres))

                if is_blocked:
                    blocked_MC_rxns.add(MC_rxn)
                    related_NC = MC_to_NC_df_dic[sample].loc[MC_rxn, :]
                    blocked_NCs_dic[sample] = set(related_NC[related_NC > 0].columns)
        MC_rxn_dic[sample] -= blocked_MC_rxns
        HC_rxn_dic[sample] |= MC_rxn_dic[sample]
    return {"blocked_NCs_dic": blocked_NCs_dic,
            "HC_rxn_dic": HC_rxn_dic,
            "MC_rxn_dic": MC_rxn_dic}


def find_HC_supp_OT_rxns(sample_names,
                         modified_model,
                         rev_rxns,
                         orig_rxns,
                         penalty_met_id,
                         penalty_rxn_id,
                         HC_rxn_dic,
                         MC_rxn_dic,
                         NC_rxn_dic,
                         OT_rxn_dic,
                         HC_score,
                         MC_score,
                         NC_score,
                         OT_score,):
    HC_rxn_dic = copy.deepcopy(HC_rxn_dic)
    OT_rxn_dic = copy.deepcopy(OT_rxn_dic)
    for sample in tqdm(sample_names):
        rxn_cost_dict = get_rxn_cost_dict(sample=sample,
                                          HC_rxn_dic=HC_rxn_dic,
                                          MC_rxn_dic=MC_rxn_dic,
                                          NC_rxn_dic=NC_rxn_dic,
                                          OT_rxn_dic=OT_rxn_dic,
                                          HC_score=HC_score,
                                          MC_score=MC_score,
                                          NC_score=NC_score,
                                          OT_score=OT_score)
        HC_to_OT_df = get_relation_df(HC_rxn_dic[sample], OT_rxn_dic[sample])
        with modified_model as model:
            for r in model.reactions:
                if r.id not in HC_rxn_dic[sample] and r.id not in OT_rxn_dic[sample]:
                    r.knock_out()

            for HC_rxn in tqdm(HC_rxn_dic[sample]):
                rxn_sets = {"OT": OT_rxn_dic[sample]}
                rxn_to_rxn_set_dfs = {"OT": HC_to_OT_df}
                model.objective = HC_rxn
                for dir in ["maximum", "minimum"]:
                    rxn = model.reactions.get_by_id(HC_rxn)
                    if (dir == "maximum" and rxn.upper_bound > 0) or (dir == "minimum" and rxn.lower_bound < 0):
                        obj_val = model.slim_optimize()
                        obj_dic = {HC_rxn: obj_val}
                        flag, supp_dict = find_supp_rxns_by_corsoFBA(model,
                                                                     sol_val=obj_val,
                                                                     sol_dic=obj_dic,
                                                                     rxn_to_check=HC_rxn,
                                                                     rxn_sets=rxn_sets,
                                                                     rxn_to_rxn_set_dfs=rxn_to_rxn_set_dfs,
                                                                     rxn_cost_dict=rxn_cost_dict,
                                                                     orig_rev_rxns=rev_rxns,
                                                                     orig_rxns=orig_rxns,
                                                                     penalty_met_id=penalty_met_id,
                                                                     penalty_rxn_id=penalty_rxn_id,
                                                                     direction=dir)
                        assert not flag, "HC reactions should be feasible in the step 7"

        OT_score = HC_to_OT_df.sum(axis=1)
        OT_score.index = HC_to_OT_df.columns
        ot_add_to_hc_rxns = set(OT_score[OT_score != 0].index)
        HC_rxn_dic[sample] |= ot_add_to_hc_rxns
        OT_rxn_dic[sample] -= ot_add_to_hc_rxns
    return {"HC_rxn_dic": HC_rxn_dic,
            "OT_rxn_dic": OT_rxn_dic}


def classify_CORDA_rxns(ref_model,
                        sample_names,
                        expression_dic,
                        used_rxn_thres):
    all_rxn_ids = set([r.id for r in ref_model.reactions])
    HC_rxn_dic, MC_rxn_dic, NC_rxn_dic, OT_rxn_dic = {}, {}, {}, {}
    for sample, exp in expression_dic.items():
        if sample in sample_names:
            HC_rxn_dic[sample] = {r for r, v in exp.rxn_scores.items()
                                  if used_rxn_thres["HC"][0] > v >= used_rxn_thres["HC"][1]}
            MC_rxn_dic[sample] = {r for r, v in exp.rxn_scores.items()
                                  if used_rxn_thres["MC"][0] > v >= used_rxn_thres["MC"][1] and
                                  r not in HC_rxn_dic[sample]}
            NC_rxn_dic[sample] = {r for r, v in exp.rxn_scores.items()
                                  if used_rxn_thres["NC"][0] > v >= used_rxn_thres["NC"][1] and
                                  r not in HC_rxn_dic[sample]}
            OT_rxn_dic[sample] = all_rxn_ids - HC_rxn_dic[sample] - MC_rxn_dic[sample] - NC_rxn_dic[sample]
    return {"HC_rxn_dic": HC_rxn_dic,
            "MC_rxn_dic": MC_rxn_dic,
            "NC_rxn_dic": NC_rxn_dic,
            "OT_rxn_dic": OT_rxn_dic}