from typing import Set, Union, List, Dict
import copy
from warnings import warn
from tqdm import tqdm

import cobra
import numpy as np
import pandas as pd

from pipeGEM.integration.mapping import Expression
from pipeGEM.integration.utils import get_rxn_set, flip_direction
from pipeGEM.utils import get_rxns_in_subsystem
from ._LP import LP3, LP7, LP9, non_convex_LP7, non_convex_LP3


def find_sparse_mode(J, P, nonP, model, singleJ, epsilon):
    if len(J) == 0:
        return []
        # print(f"find_sparse_mode of single reaction: {singleJ}")
    supps, v = LP7(J if singleJ is None else singleJ,
                   model, epsilon, use_abs=False, return_min_v=True)
    K = np.intersect1d(list(J), supps)
    if singleJ is not None and len(singleJ & set(K)) == 0:
        warn(f"Singleton {singleJ} flux cannot be generated in LP7")
    if K.shape[0] == 0:
        return []
    return LP9(K, P, nonP, model, epsilon, min_v=v[list(K)].min())


def fastcc(model,
           epsilon,
           return_model: bool,
           return_rxn_ids: bool,
           return_removed_rxn_ids: bool,
           is_convex=True) -> dict:
    if not is_convex:
        print("Using non-convex fastcc method")
    print(f"Epsilon used: {epsilon}")
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
        with tqdm(total=len(J)) as pbar:
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
                pbar.update(before_n - after_n)
                if before_n != after_n:
                    flipped = False
                else :  # no change in number of rxn_to_keeps
                    Jirev = Ji - irr_rxns
                    if flipped or len(Jirev) == 0:
                        flipped = False
                        if singleton:
                            J -= Ji
                            # print("[Removed] ", Ji, "is flux inconsistent.")
                            pbar.update(1)
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
             return_removed_rxn_ids: bool,
             raise_err: bool = False,
             scale_by_coef: bool = True):
    if return_model:
        output_model = model.copy()
    if not isinstance(C, set):
        C = set(C)
    if not isinstance(nonP, set):
        nonP = set(nonP)
    # if scale_by_coef:
    #     print("Use scaled epsilons")
    #     rxn_scale_eps = pd.Series({r.id: 0.99*epsilon / max([abs(mc) for mc in r.metabolites.values()])
    #                                for r in model.reactions})
    # else:
    #     rxn_scale_eps = None

    all_rxns = get_rxn_set(model)
    irr_rxns = get_rxn_set(model, "irreversible")
    backward_rxns = get_rxn_set(model, "backward")

    with model:
        if len(backward_rxns) > 0:
            flip_direction(model, backward_rxns)

        flipped, singleton = False, False
        J = C & irr_rxns
        P = all_rxns - C - nonP
        singleJ = None
        A = set(find_sparse_mode(J, P, nonP, model, singleJ, epsilon))
        invalid_part = J - A
        track_irrev = False
        if len(invalid_part) != 0:
            track_irrev = True
            warn(f"Inconsistent irreversible core reactions (They should be included in A): Total: {len(invalid_part)}")
        J = C - A  # reactions to be added to the model
        with tqdm(total=len(J)) as pbar:
            n_j = len(J)

            while len(J) > 0:
                P = P - A
                supp = set(find_sparse_mode(J, P, nonP, model, singleJ, epsilon))
                A |= supp
                if len(J & A) > 0:
                    J -= A
                    pbar.update(n_j - len(J))
                    n_j = len(J)
                    flipped, singleton = False, False
                    singleJ = None

                    if track_irrev:
                        print(f"Irrev in J: {len(J & irr_rxns)}, J: {len(J)}")
                else:
                    if singleton and len(J - irr_rxns) != 0:
                        if singleJ is None:
                            Jrev = {next(iter(J - irr_rxns))}
                            singleJ = Jrev
                            flipped = False
                    else:
                        Jrev = J - irr_rxns
                    if len(Jrev) == 0 or flipped:  # If no reversible J or the model is flipped
                        if singleton:
                            if raise_err:
                                raise ValueError(f"Error: Global network is not consistent. Last rxn: {J} |J| = {len(J)}")
                            else:
                                to_remove = {next(iter(J))}
                                warn(f"Error: Global network is not consistent. Removed core rxn: {to_remove}")
                                J -= to_remove
                                n_j = len(J)
                                pbar.update(1)
                                flipped, singleton = False, False
                                singleJ = None
                        else:
                            flipped, singleton = False, True
                            if singleJ is None and len(J - irr_rxns) != 0:
                                Jrev = {next(iter(J - irr_rxns))}
                                singleJ = Jrev
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


def supp_protected_rxns_one_sample(ref_model,
                                   name,
                                   protected_rxns,
                                   C,
                                   P,
                                   epsilon_for_fastcore,
                                   predefined_protected_rxns = None):
    if predefined_protected_rxns is not None:
        protected_rxns = predefined_protected_rxns
    C = copy.deepcopy(C)
    P = copy.deepcopy(P)
    supp = set()
    if protected_rxns is not None:
        supp = fastCore(protected_rxns,
                        C,
                        ref_model,
                        epsilon_for_fastcore,
                        return_model=False,
                        return_rxn_ids=True,
                        return_removed_rxn_ids=False)["rxn_ids"]
        print("Find", len(supp), "supporting reactions for protected rxns in ", name)
        n_before_add, n_before_subtract = len(C), len(P)
        C = C | supp
        P = P - supp
        n_after_add, n_after_subtract = len(C), len(P)
        print(n_after_add - n_before_add, "reactions added to Core reactions")
        print(n_before_subtract - n_after_subtract, "reactions remove from non-Core reactions")
    return C, P, supp


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
            result = supp_protected_rxns_one_sample(ref_model,
                                                    sample_name,
                                                    protected_rxns[
                                                        sample_name],
                                                    C_dic[
                                                        sample_name],
                                                    P_dic[
                                                        sample_name],
                                                    epsilon_for_fastcore,
                                                    None)
            C_dic[sample_name], P_dic[sample_name], supp_dic[sample_name] = result
    return {"C_dic": C_dic,
            "P_dic": P_dic,
            "supp_dic": supp_dic}


def get_unpenalized_rxn_ids_one_sample(ref_model,
                                       C, P,
                                       unpenalized_subsystem):
    unpenalized_rxn = set()
    if unpenalized_subsystem:
        unpenalized_rxn = set(get_rxns_in_subsystem(ref_model, unpenalized_subsystem)) - C
    return unpenalized_rxn, P - unpenalized_rxn


def get_unpenalized_rxn_ids(ref_model,
                            C_dic,
                            P_dic,
                            sample_names,
                            unpenalized_subsystem):
    unpenalized_rxn_dic = {}
    P_dic = copy.deepcopy(P_dic)
    if unpenalized_subsystem:
        unpenalized_rxn_ids = set(get_rxns_in_subsystem(ref_model, unpenalized_subsystem))
    else:
        unpenalized_rxn_ids = set()
    for i, sample_name in enumerate(sample_names):
        unpenalized_rxn_dic[sample_name] = (unpenalized_rxn_ids - C_dic[sample_name])
        P_dic[sample_name] -= unpenalized_rxn_dic[sample_name]

    return {"unpenalized_rxn_dic": unpenalized_rxn_dic,
            "P_dic": P_dic}


def get_cons_p_free_mod_one_sample(ref_model,
                                   C,
                                   P,
                                   epsilon_for_fastcc):
    with ref_model as mod:
        for rxn_id in P:
            mod.reactions.get_by_id(rxn_id).bounds = (0, 0)
        fastcc_out = fastcc(mod,
                            epsilon_for_fastcc,
                            return_model=True,
                            return_rxn_ids=True,
                            return_removed_rxn_ids=True,
                            is_convex=True)
    return fastcc_out["model"], fastcc_out["rxn_ids"] & C, fastcc_out["removed_rxn_ids"]


def get_consistent_p_free_model(ref_model,
                                sample_names,
                                C_dic,
                                P_dic,
                                epsilon_for_fastcc):
    p_free_model_dic = {}
    C_dic = copy.deepcopy(C_dic)
    removed_rxn_id_dic = {}
    for i, sample_name in enumerate(sample_names):
        result = get_cons_p_free_mod_one_sample(ref_model, C_dic[sample_name], P_dic[sample_name], epsilon_for_fastcc)
        p_free_model_dic[sample_name] = result[0]
        C_dic[sample_name] = result[1]
        removed_rxn_id_dic[sample_name] = result[2]
    return {"removed_rxn_id_dic": removed_rxn_id_dic,
            "C_dic": C_dic,
            "p_free_model_dic": p_free_model_dic}


def get_C_and_P_dic(expression_dic: Dict[str, Expression],  # use discrete data
                    sample_names: List[str],
                    consensus_proportion: float,
                    is_generic_model: bool):
    assert consensus_proportion > 0
    consensus_thres = consensus_proportion * (len(sample_names) if is_generic_model else 1)
    C_dic, P_dic = {}, {}
    exp_df = pd.DataFrame({sample_name: expression_dic[sample_name].rxn_scores for sample_name in sample_names}).dropna(how="all")
    output_names = ["generic"] if is_generic_model else exp_df.columns
    if is_generic_model:
        exp_df["generic"] = exp_df.mean(axis=1)
    for sample_name in output_names:
        C_dic[sample_name] = set(exp_df[exp_df[sample_name] >= consensus_thres].index)
        P_dic[sample_name] = set(exp_df[exp_df[sample_name] <= -consensus_thres].index)
    return {"C_dic": C_dic, "P_dic": P_dic}


def get_final_model(ref_model,
                    C, P,
                    unpenalized_rxn,
                    p_free_model,
                    epsilon_for_fastcore
                    ):
    with ref_model as mod:
        for r_name in P:
            mod.reactions.get_by_id(r_name).bounds = (0, 0)
        result_dic = fastCore(C,
                              unpenalized_rxn,
                              p_free_model,
                              epsilon_for_fastcore,
                              return_model=True,
                              return_rxn_ids=False,
                              return_removed_rxn_ids=True)
    return result_dic["model"]


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
        result_dic = get_final_model(ref_model,
                        C_dic[sample_name], P_dic[sample_name],
                        unpenalized_rxn_dic[sample_name],
                        p_free_model_dic[sample_name],
                        epsilon_for_fastcore)
        final_model_dic[sample_name] = result_dic["model"]
        removed_rxn_id_dic[sample_name] = result_dic["removed_rxn_ids"]

    return {"final_model_dic": final_model_dic,
            "removed_rxn_id_dic": removed_rxn_id_dic}