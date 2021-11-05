from typing import Set, Union, List

import cobra
import numpy as np

from pipeGEM.integration.utils import get_rxn_set, flip_direction
from ._LP import LP3, LP7, LP9, non_convex_LP7, non_convex_LP3


def find_sparse_mode(J, P, nonP, model, singleton, epsilon):
    if len(J) == 0:
        return []
    if singleton:
        singleJ = {next(iter(J))}
        # print(f"find_sparse_mode of single reaction: {singleJ}")
    K = np.intersect1d(list(J), (LP7(J if not singleton else singleJ, model, epsilon, use_abs=False)))
    if K.shape[0] == 0:
        return []
    return LP9(K, P, nonP, model, epsilon)


def fastcc(model,
           epsilon,
           return_model: bool,
           return_rxn_ids: bool,
           return_removed_rxn_ids: bool,
           is_convex=True) -> dict:
    if return_model:
        consistent_model = model.copy()
    all_rxns = get_rxn_set(model)
    irr_rxns = get_rxn_set(model, "irreversible")
    no_expressed = get_rxn_set(model, "not_expressed")
    backward_rxns = get_rxn_set(model, "backward")
    J = irr_rxns - no_expressed
    with model:
        if len(backward_rxns) > 0:
            print(f"Found and flipped {len(backward_rxns)} reactions")
            flip_direction(model, backward_rxns)

        A = set(LP7(J, model, epsilon, use_abs=True))  # rxns to keeps
        # print("A: ", len(A))
        J = all_rxns - A - J - no_expressed  # rev rxns to check
        # print("J: ", len(J))
        singleton, flipped = False, False

        while len(J) != 0:
            if singleton:
                Ji = {next(iter(J))}
                new_supps = set(LP3(Ji, model, epsilon)) if is_convex else set(non_convex_LP3(Ji, model, epsilon))
            else:
                Ji = J.copy()
                new_supps = set(LP7(Ji, model, epsilon)) if is_convex else set(non_convex_LP7(Ji, model, epsilon))
            A |= new_supps
            before_n = len(J)
            J -= A
            after_n = len(J)
            if before_n != after_n:
                flipped = False
            else :  # no change in number of rxn_to_keeps
                Jirev = Ji - irr_rxns
                if flipped or len(Jirev) == 0:
                    flipped = False
                    if singleton:
                        J -= Ji
                        print("[Removed] ", Ji, "is flux inconsistent.")
                    else:
                        singleton = True
                else:
                    flip_direction(model, Jirev)
                    flipped = True
    rxns_to_remove = list(all_rxns - A)
    output = {}
    if return_model:
        consistent_model.remove_reactions(rxns_to_remove, remove_orphans=True)
        output["model"] = consistent_model

    if return_removed_rxn_ids:
        output["removed_rxn_ids"] = rxns_to_remove

    if return_rxn_ids:
        output["rxn_ids"] = A

    return output


def fastCore(C: Union[List[str], Set[str]],
             nonP: Union[List[str], Set[str]],
             model: cobra.Model,
             epsilon: float,
             return_model: bool,
             return_rxn_ids: bool,
             return_removed_rxn_ids: bool):
    if return_model:
        output_model = model.copy()
    if not isinstance(C, set):
        C = set(C)
    if not isinstance(nonP, set):
        nonP = set(nonP)

    all_rxns = get_rxn_set(model)
    irr_rxns = get_rxn_set(model, "irreversible")
    backward_rxns = get_rxn_set(model, "backward")

    with model:
        if len(backward_rxns) > 0:
            flip_direction(model, backward_rxns)

        flipped, singleton = False, False
        J = C & irr_rxns
        P = all_rxns - C - nonP
        A = set(find_sparse_mode(J, P, nonP, model, singleton, epsilon))
        invalid_part = J - A
        assert len(invalid_part) == 0, \
            f"Inconsistent irreversible core reactions (They should be included in A): {invalid_part}"
        J = C - A  # reactions to be added to the model
        while len(J) > 0:
            P = P - A
            supp = set(find_sparse_mode(J, P, nonP, model, singleton, epsilon))
            A |= supp
            if len(J & A) > 0:
                J -= A
                flipped = False
            else:
                Jrev = {next(iter(J))} - irr_rxns if singleton else J - irr_rxns
                if len(Jrev) == 0 or flipped:  # If no reversible J or the model is flipped
                    assert not singleton, "Error: Global network is not consistent."
                    flipped, singleton = False, True
                else:
                    flip_direction(model, Jrev)
                    flipped = True
    rxns_to_remove = list(all_rxns - A)
    output = {}
    if return_model:
        output_model.remove_reactions(rxns_to_remove, remove_orphans=True)
        output["model"] = output_model

    if return_removed_rxn_ids:
        output["removed_rxn_ids"] = rxns_to_remove

    if return_rxn_ids:
        output["rxn_ids"] = A
    return output


def supp_protected_rxns(ref_model,
                        sample_names,
                        protected_rxns,
                        C_dic,
                        P_dic,
                        epsilon_for_fastcore,
                        predefined_protected_rxns = None):
    if predefined_protected_rxns is not None:
        protected_rxns = predefined_protected_rxns

    C_dic = copy.deepcopy(C_dic)
    P_dic = copy.deepcopy(P_dic)
    supp_dic = {}  # for recording
    for i, sample_name in enumerate(sample_names):
        if protected_rxns[sample_name]:
            supp = fastCore(protected_rxns[sample_name],
                            C_dic[sample_name],
                            ref_model,
                            epsilon_for_fastcore,
                            return_model=False,
                            return_rxn_ids=True,
                            return_removed_rxn_ids=False)["rxn_ids"]
            print("Find", len(supp), "supporting reactions for protected rxns in ", sample_name)
            n_before_add, n_before_subtract = len(C_dic[sample_name]), len(P_dic[sample_name])
            C_dic[sample_name] = C_dic[sample_name] | supp
            P_dic[sample_name] = P_dic[sample_name] - supp
            n_after_add, n_after_subtract = len(C_dic[sample_name]), len(P_dic[sample_name])
            supp_dic[sample_name] = supp
            print(n_after_add - n_before_add, "reactions added to Core reactions")
            print(n_before_subtract - n_after_subtract, "reactions remove from non-Core reactions")
    return {"C_dic": C_dic,
            "P_dic": P_dic,
            "supp_dic": supp_dic}


def get_unpenalized_rxn_ids(ref_model,
                            C_dic,
                            P_dic,
                            sample_names,
                            unpenalized_subsystem):
    unpenalized_rxn_dic = {}
    P_dic = copy.deepcopy(P_dic)
    for i, sample_name in enumerate(sample_names):
        if unpenalized_subsystem:
            patterns = [re.compile(sub) for sub in unpenalized_subsystem]
            unpenalized_rxn_ids = []
            for rxn in ref_model.reactions:
                if rxn.id not in C_dic[sample_name] and any([pat.match(rxn.subsystem) for pat in patterns]):
                    unpenalized_rxn_ids.append(rxn.id)
            unpenalized_rxn_dic[sample_name] = set(unpenalized_rxn_ids)
        else:
            unpenalized_rxn_dic[sample_name] = set()
        P_dic[sample_name] -= unpenalized_rxn_dic[sample_name]

    return {"unpenalized_rxn_dic": unpenalized_rxn_dic,
            "P_dic": P_dic}


def get_consistent_p_free_model(ref_model,
                                sample_names,
                                C_dic,
                                P_dic,
                                epsilon_for_fastcc):
    p_free_model_dic = {}
    C_dic = copy.deepcopy(C_dic)
    C_dic_added_rxn_dic = {}  # for recording
    removed_rxn_id_dic = {}
    for i, sample_name in enumerate(sample_names):
        with ref_model as mod:
            for rxn_id in P_dic[sample_name]:
                mod.reactions.get_by_id(rxn_id).bounds = (0, 0)
            fastcc_out = fastcc(mod,
                                epsilon_for_fastcc,
                                return_model=True,
                                return_rxn_ids=True,
                                return_removed_rxn_ids=True,
                                is_convex=True)
            new_consist_A, p_free_model_dic[sample_name] = fastcc_out["rxn_ids"], fastcc_out["model"]
            C_dic_added_rxn_dic[sample_name] = new_consist_A - C_dic[sample_name]
            C_dic[sample_name] = C_dic[sample_name] & new_consist_A
            removed_rxn_id_dic[sample_name] = fastcc_out["removed_rxn_ids"]
    return {"removed_rxn_id_dic": removed_rxn_id_dic,
            "C_dic_added_rxn_dic": C_dic_added_rxn_dic,
            "C_dic": C_dic,
            "p_free_model_dic": p_free_model_dic}


def get_C_and_P_dic(expression_dic: Dict[str, Expression],  # use discrete data
                    sample_names: List[str],
                    consensus_proportion: float,
                    is_generic_model: bool):
    assert consensus_proportion > 0
    consensus_thres = consensus_proportion * (len(sample_names) if is_generic_model else 1)
    C_dic, P_dic = {}, {}
    for sample_name in sample_names:
        C_dic[sample_name] = {rxn for rxn, score in expression_dic[sample_name].rxn_scores.items()
                              if score >= consensus_thres}
        P_dic[sample_name] = {rxn for rxn, score in expression_dic[sample_name].rxn_scores.items()
                              if score <= -consensus_thres}
    return {"C_dic": C_dic, "P_dic": P_dic}


def get_final_models(ref_model,
                     sample_names,
                     C_dic,
                     P_dic,
                     unpenalized_rxn_dic,
                     p_free_model_dic,
                     epsilon_for_fastcore):
    final_model_dic = {}
    removed_rxn_id_dic = {}
    for i, sample_name in enumerate(sample_names):
        with ref_model as mod:
            for r_name in P_dic[sample_name]:
                mod.reactions.get_by_id(r_name).bounds = (0, 0)
            result_dic = fastCore(C_dic[sample_name],
                                  unpenalized_rxn_dic[sample_name],
                                  p_free_model_dic[sample_name],
                                  epsilon_for_fastcore,
                                  return_model=True,
                                  return_rxn_ids=False,
                                  return_removed_rxn_ids=True)
            final_model_dic[sample_name] = result_dic["model"]
            removed_rxn_id_dic[sample_name] = result_dic["removed_rxn_ids"]
    return {"final_model_dic": final_model_dic,
            "removed_rxn_id_dic": removed_rxn_id_dic}