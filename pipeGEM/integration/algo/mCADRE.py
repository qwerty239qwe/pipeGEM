import numpy as np
import pandas as pd
from cobra.util.array import create_stoichiometric_matrix
from pipeGEM.analysis.tasks import Task, TaskContainer, TaskHandler
from pipeGEM.analysis import consistency_testers, timing, mCADRE_Analysis
from tqdm import tqdm


def calc_expr_score(data, expr_th, nonexpr_th, absent_value, absent_value_indicator=-1e-6) -> dict:
    rxn_ids, rxn_scores = list(data.rxn_scores.keys()), list(data.rxn_scores.values())
    mapped_scores = np.interp(rxn_scores, [nonexpr_th, expr_th], [0, 1])
    mapped_scores[rxn_scores == absent_value] = absent_value_indicator
    return dict(zip(rxn_ids, mapped_scores))


def calc_corr_score(model, expr_scores) -> dict:
    s = create_stoichiometric_matrix(model)
    con_mat = s.T @ s
    con_mat[con_mat != 0] = 1
    con_mat = con_mat - np.eye(con_mat.shape[0])
    edges = con_mat.sum(axis=1)
    edge_mat = np.tile(edges, con_mat.shaep[0])
    expr_mat = np.tile(np.array([expr_scores[r.id] for r in model.reactions]),
                       con_mat.shape[0])
    con_mat = con_mat * expr_mat / edge_mat
    return dict(zip([r.id for r in model.reactions], con_mat.sum(axis=0)))


def get_score_df(model,
                 data,
                 expr_th,
                 nonexpr_th,
                 exp_cutoff=0.9,
                 absent_value: float = 0,
                 absent_value_indicator=-1e-6,
                 evidence_scores=None):
    expr_scores = calc_expr_score(data, expr_th, nonexpr_th, absent_value, absent_value_indicator)
    core_rxns = [r for r, exp in expr_scores.items() if exp >= exp_cutoff]
    conn_scores = calc_corr_score(model, expr_scores)
    non_expressed_rxns = [r for r, exp in expr_scores.items() if exp <= absent_value_indicator]
    if evidence_scores is None:
        evidence_scores = {r.id: 0 for r in model.reactions}
    elif isinstance(evidence_scores, dict):
        es_not_included = {r.id: 0 for r in model.reactions if r.id not in evidence_scores}
        evidence_scores = {**evidence_scores, **es_not_included}
    score_df = pd.DataFrame({"expression": expr_scores,
                             "connectivity": conn_scores,
                             "evidence": evidence_scores}).sort_values(["expression", "connectivity", "evidence"])
    return score_df, core_rxns, non_expressed_rxns


def _get_mCADRE_default_func_test_tasks(required_met_ids,
                                        glucose_m_id = "MAM01965",
                                        co2_m_id = "MAM01596",
                                        cytosol_comp = "c",
                                        extracellular_comp = "e"):
    tasks = TaskContainer()
    in_mets = [{"met_id": glucose_m_id, "lb": -1000, "ub": 5, "compartment": extracellular_comp},
               {"met_id": co2_m_id, "lb": -1000, "ub": 1000, "compartment": extracellular_comp}]
    for met in required_met_ids:
        tasks[f"met_{met}"] = Task(should_fail=False,
                                   in_mets=in_mets,
                                   out_mets=[{"met_id": met, "lb": 0, "ub": 1000, "compartment": cytosol_comp}],
                                   ko_input_type="organic", ko_output_type="organic")
    return tasks


def _get_mCADRE_default_salvage_test_tasks(gmp_m_id = "MAM02016",
                                           guanine_m_id = "MAM02037",
                                           imp_m_id = "MAM02167",
                                           hypoxanthine_m_id = "MAM02159",
                                           cytosol_comp = "c",
                                           extracellular_comp = "e"):
    tasks = TaskContainer()
    tasks[f"guanine_to_gmp"] = Task(should_fail=False,
                                    in_mets=[{"met_id": guanine_m_id, "lb": 0, "ub": 5,
                                              "compartment": extracellular_comp},],
                                    out_mets=[{"met_id": gmp_m_id, "lb": 0, "ub": 1e-6, "compartment": cytosol_comp}],
                                    ko_input_type="organic", ko_output_type="organic")
    tasks[f"guanine_to_gmp"] = Task(should_fail=False,
                                    in_mets=[{"met_id": hypoxanthine_m_id, "lb": 0, "ub": 5,
                                              "compartment": extracellular_comp}, ],
                                    out_mets=[{"met_id": imp_m_id, "lb": 0, "ub": 1e-6, "compartment": cytosol_comp}],
                                    ko_input_type="organic", ko_output_type="organic")
    return tasks


def _get_func_test_result(model,
                          func_test_tasks=None,
                          required_met_ids=None,
                          default_func_test=False,):
    func_task_handler, func_test_result = None, None
    if required_met_ids is not None:
        if default_func_test:
            tasks = _get_mCADRE_default_func_test_tasks(required_met_ids)
            func_task_handler = TaskHandler(model=model, tasks_path_or_container=tasks)
    elif func_test_tasks is not None:
        func_task_handler = TaskHandler(model=model, tasks_path_or_container=func_test_tasks)

    if func_task_handler is not None:
        func_test_result = func_task_handler.test_tasks(n_additional_path=100)
    return func_test_result


def _get_salv_test_result(model,
                          salvage_check_tasks=None,
                          default_salv_test=False,
                          **kwargs):
    salvage_task_handler, salvage_test_result = None, None
    if default_salv_test:
        tasks = _get_mCADRE_default_salvage_test_tasks(**kwargs)
        salvage_task_handler = TaskHandler(model=model, tasks_path_or_container=tasks)
    elif salvage_check_tasks is not None:
        salvage_task_handler = TaskHandler(model=model, tasks_path_or_container=salvage_check_tasks)

    if salvage_task_handler is not None:
        salvage_test_result = salvage_task_handler.test_tasks(n_additional_path=100)
    return salvage_test_result


def _check_all_test(rxn_id, func_test_result, salv_test_result):
    pass_func_test, pass_salv_test = True, True
    if func_test_result is not None:
        pass_func_test = not func_test_result.is_essential(rxn_id)

    if salv_test_result is not None:
        pass_salv_test = not salv_test_result.is_essential(rxn_id)

    return pass_salv_test and pass_func_test


def _prune_model(model,
                 score_df,
                 core_rxns,
                 non_expressed_rxns,
                 func_test_result,
                 salv_test_result,
                 eta=0.333,
                 consistency_test_method="FASTCC",
                 tolerance=1e-8,
                 ):
    non_core_df = score_df.loc[~score_df.index.isin(core_rxns), :]
    all_removed_rxn_ids = []

    for non_core_rxn_id in tqdm(non_core_df.index):
        if non_core_rxn_id in all_removed_rxn_ids:
            continue

        pass_tests = _check_all_test(non_core_rxn_id, func_test_result, salv_test_result)
        if not pass_tests:
            continue

        with model:
            model.reactions.get_by_id(non_core_rxn_id).bounds = (0, 0)
            #consistency check
            cons_tester = consistency_testers[consistency_test_method](model)
            test_result = cons_tester.analyze(tol=tolerance,
                                              return_model=False)
            removed_rxns = test_result.removed_rxn_ids
            pruned_model = test_result.consistent_model

        pass_tests = all(_check_all_test(rid, func_test_result, salv_test_result) for rid in removed_rxns)
        if not pass_tests:
            continue

        n_core_rm = len(set(core_rxns) & set(removed_rxns))
        n_non_core_rm = len(set(non_core_df.index) & set(removed_rxns))

        if (non_core_rxn_id in non_expressed_rxns and (n_core_rm <= n_non_core_rm * eta)) or (n_core_rm == 0):
            model = pruned_model
            all_removed_rxn_ids.extend(removed_rxns)

    return model, all_removed_rxn_ids


@timing
def apply_mCADRE(model,
                 data,
                 expr_th,
                 nonexpr_th,
                 exp_cutoff: float = 0.9,
                 absent_value: float = 0,
                 absent_value_indicator: float = -1e-6,
                 evidence_scores=None,
                 salvage_check_tasks=None,
                 default_salv_test=False,
                 func_test_tasks=None,
                 required_met_ids=None,
                 default_func_test=False,
                 ) -> mCADRE_Analysis:
    score_df, core_rxns, non_expressed_rxns = get_score_df(model,
                                                           data,
                                                           expr_th,
                                                           nonexpr_th,
                                                           exp_cutoff=exp_cutoff,
                                                           absent_value=absent_value,
                                                           absent_value_indicator=absent_value_indicator,
                                                           evidence_scores=evidence_scores)

    func_test_result = _get_func_test_result(model=model,
                                             func_test_tasks=func_test_tasks,
                                             required_met_ids=required_met_ids,
                                             default_func_test=default_func_test,)
    if func_test_result is not None and (not func_test_result.result_df["Passed"].all()):
        raise RuntimeError(f"The input model could not perform all provided metabolic tests,"
                           f"Test details: {func_test_result.result_df}")

    salvage_test_result = _get_salv_test_result(model=model,
                                                salvage_check_tasks=salvage_check_tasks,
                                                default_salv_test=default_salv_test,)

    if salvage_test_result is not None and (not salvage_test_result.result_df["Passed"].all()):
        raise RuntimeError(f"The input model could not perform all provided salvage tests,"
                           f"Test details: {salvage_test_result.result_df}")

    result_model, removed_rxn_ids = _prune_model(model,
                                                    score_df,
                                                    core_rxns,
                                                    non_expressed_rxns,
                                                    func_test_result,
                                                    salvage_test_result,
                                                    eta=0.333,
                                                    consistency_test_method="FASTCC",
                                                    tolerance=1e-8,)
    result = mCADRE_Analysis(log=dict(exp_cutoff=exp_cutoff,
                                      absent_value=absent_value,
                                      absent_value_indicator=absent_value_indicator,
                                      evidence_scores=evidence_scores,))

    result.add_result(model=result_model,
                      removed_rxns=removed_rxn_ids,
                      core_rxns=core_rxns,
                      score_df=score_df,
                      non_expressed_rxns=non_expressed_rxns,
                      salvage_test_result=salvage_test_result,
                      func_test_result=func_test_result)
    return result