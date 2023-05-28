import pandas as pd
import copy
import pandas as pd


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
                                                    protected_rxns[sample_name],
                                                    C_dic[sample_name],
                                                    P_dic[sample_name],
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
    C_dic, P_dic = {}, {}
    exp_df = pd.DataFrame({sample_name: expression_dic[sample_name].rxn_scores for sample_name in sample_names}).dropna(how="all")
    output_names = ["generic"] if is_generic_model else exp_df.columns
    if is_generic_model:
        exp_df["generic"] = exp_df.mean(axis=1)
    for sample_name in output_names:
        C_dic[sample_name] = set(exp_df[exp_df[sample_name] >= consensus_proportion].index)
        P_dic[sample_name] = set(exp_df[exp_df[sample_name] <= -consensus_proportion].index)
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