from typing import Union, Dict

import numpy as np
import pandas as pd
from cobra.util.array import create_stoichiometric_matrix
from cobra.exceptions import OptimizationError

from pipeGEM.analysis.tasks import Task, TaskContainer, TaskHandler
from pipeGEM.analysis import consistency_testers, timing, mCADRE_Analysis, \
    NumInequalityStoppingCriteria, IsInSetStoppingCriteria, measure_efficacy
from pipeGEM.integration.utils import parse_predefined_threshold
from tqdm import tqdm


def calc_expr_score(data,
                    expr_th,
                    nonexpr_th,
                    absent_value,
                    protected_rxns=None,
                    absent_value_indicator=-1e-6) -> dict:
    if protected_rxns is None:
        protected_rxns = []

    rxn_ids, rxn_scores = list(data.rxn_scores.keys()), list(data.rxn_scores.values())
    mapped_scores = np.interp(rxn_scores, [nonexpr_th, expr_th], [0, 1])
    mapped_scores[rxn_scores == absent_value] = absent_value_indicator
    expr_scores = dict(zip(rxn_ids, mapped_scores))
    for r_id in expr_scores.keys():
        if r_id in protected_rxns:
            expr_scores[r_id] = 1

    return expr_scores


def calc_corr_score(model, expr_scores) -> dict:
    s = create_stoichiometric_matrix(model)
    con_mat = s.T @ s  # R x R
    con_mat[con_mat != 0] = 1
    con_mat = con_mat - np.eye(con_mat.shape[0])
    edges = con_mat.sum(axis=1)
    edge_mat = np.tile(edges, (con_mat.shape[0], 1))
    expr_mat = np.tile(np.array([expr_scores[r.id] for r in model.reactions]),
                       (con_mat.shape[0], 1))
    con_mat = con_mat * expr_mat / edge_mat
    con_mat[~np.isfinite(con_mat)] = 0
    return dict(zip([r.id for r in model.reactions], con_mat.sum(axis=1)))


def get_score_df(model,
                 data,
                 expr_th,
                 nonexpr_th,
                 protected_rxns=None,
                 exp_cutoff=0.9,
                 absent_value: float = 0,
                 absent_value_indicator=-1e-6,
                 evidence_scores=None):
    expr_scores = calc_expr_score(data,
                                  expr_th, nonexpr_th,
                                  absent_value,
                                  protected_rxns=protected_rxns,
                                  absent_value_indicator=absent_value_indicator)
    core_rxns = [r for r, exp in expr_scores.items() if exp >= exp_cutoff]
    conn_scores = calc_corr_score(model, expr_scores)
    non_expressed_rxns = [r for r, exp in expr_scores.items() if exp <= absent_value_indicator]
    if evidence_scores is None:
        evidence_scores = {r.id: 0 for r in model.reactions}
    elif isinstance(evidence_scores, dict):
        es_not_included = {r.id: 0 for r in model.reactions if r.id not in evidence_scores}
        evidence_scores = {**evidence_scores, **es_not_included}
    elif isinstance(evidence_scores, pd.Series):
        evidence_scores = evidence_scores.to_dict()
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
                 protected_rxns,
                 func_test_result,
                 salv_test_result,
                 eta=0.333,
                 consistency_test_method="FASTCC",
                 tolerance=1e-8,
                 rxn_scaling_coefs=None):
    non_core_df = score_df.loc[~score_df.index.isin(core_rxns), :]
    all_removed_rxn_ids = []
    model = model.copy()

    for non_core_rxn_id in tqdm(non_core_df.index):
        if non_core_rxn_id in all_removed_rxn_ids:
            continue

        pass_tests = _check_all_test(non_core_rxn_id, func_test_result, salv_test_result)
        if not pass_tests:
            continue

        if non_core_rxn_id in non_expressed_rxns:
            stop_crit = [IsInSetStoppingCriteria(ess_set=set(core_rxns) - set(all_removed_rxn_ids))]
        else:
            stop_crit = [NumInequalityStoppingCriteria(var={"core": list(set(core_rxns) - set(all_removed_rxn_ids)),
                                                            "non_core": list(set(non_core_df.index) -
                                                                             set(all_removed_rxn_ids))},
                                                       cons_dict={"core": -1,
                                                                  "non_core": eta})]
        stop_crit.append(IsInSetStoppingCriteria(ess_set=set(protected_rxns)))

        cons_tester = consistency_testers[consistency_test_method](model)
        with model:
            tqdm.write(f"Trying to remove {non_core_rxn_id}")
            model.reactions.get_by_id(non_core_rxn_id).bounds = (0, 0)
            #consistency check
            try:
                if consistency_test_method == "FASTCC":
                    test_result = cons_tester.analyze(tol=tolerance,
                                                      return_model=False,
                                                      stopping_callback=stop_crit,
                                                      rxn_scaling_coefs=rxn_scaling_coefs)
                else:
                    test_result = cons_tester.analyze(tol=tolerance,
                                                      return_model=False,
                                                      rxn_scaling_coefs=rxn_scaling_coefs)
            except OptimizationError:
                print(f"{non_core_rxn_id} knocking-out makes the model infeasible.")
                continue
        if "stopped" in test_result.log and test_result.log["stopped"]:
            continue
        removed_rxns = test_result.removed_rxn_ids
        pass_tests = all(_check_all_test(rid, func_test_result, salv_test_result) for rid in removed_rxns)
        if not pass_tests:
            continue
        for r in removed_rxns:
            model.reactions.get_by_id(r).bounds = (0, 0)
        all_removed_rxn_ids.extend(removed_rxns)
    all_removed_rxn_ids = np.array(list(set(all_removed_rxn_ids)))
    model.remove_reactions(all_removed_rxn_ids)
    return model, all_removed_rxn_ids


@timing
def apply_mCADRE(model,
                 data,
                 protected_rxns,
                 predefined_threshold = None,
                 threshold_kws: dict = None,
                 rxn_scaling_coefs: dict = None,
                 exp_cutoff: float = 0.9,
                 absent_value: float = 0,
                 absent_value_indicator: float = -1e-6,
                 tol=1e-6,
                 eta=0.333,
                 evidence_scores: Union[Dict[str, Union[int, float]], pd.Series] = None,
                 salvage_check_tasks=None,
                 default_salv_test=False,
                 func_test_tasks=None,
                 required_met_ids=None,
                 default_func_test=False,
                 ) -> mCADRE_Analysis:
    """Apply the mCADRE algorithm to generate a context-specific metabolic model.

    mCADRE (metabolic Context-specificity Assessed by Deterministic Reaction Evaluation)
    builds context-specific models by iteratively removing reactions based on expression
    data, network connectivity, and optional evidence scores, while ensuring the model
    can still perform essential metabolic functions (tasks).

    Parameters
    ----------
    model : cobra.Model
        The input genome-scale metabolic model.
    data : object
        An object containing gene expression data (`data.gene_data`) and
        reaction scores (`data.rxn_scores`) derived from it.
    protected_rxns : list[str]
        A list of reaction IDs that should never be removed from the model.
    predefined_threshold : dict or analysis_types, optional
        Strategy or dictionary defining thresholds (`exp_th`, `non_exp_th`) to
        classify reactions based on scores. See
        `pipeGEM.integration.utils.parse_predefined_threshold`. Defaults to None.
    threshold_kws : dict, optional
        Additional keyword arguments for the thresholding function. Defaults to None.
    rxn_scaling_coefs : dict[str, float], optional
        Dictionary mapping reaction IDs to scaling coefficients, used to adjust
        consistency check tolerance. Defaults to None.
    exp_cutoff : float, optional
        Expression score threshold (after mapping to [0, 1]) above which a
        reaction is considered part of the initial 'core' set. Defaults to 0.9.
    absent_value : float, optional
        The raw score in `data.rxn_scores` that indicates a reaction is absent
        (e.g., 0 for some expression data types). Defaults to 0.
    absent_value_indicator : float, optional
        The internal score assigned to absent reactions after mapping. Should be
        less than 0. Defaults to -1e-6.
    tol : float, optional
        Tolerance used for consistency checks (e.g., FASTCC). Defaults to 1e-6.
    eta : float, optional
        Weighting factor used in the consistency check stopping criteria when
        evaluating removal of medium-confidence reactions. Represents the
        trade-off between removing non-core reactions and keeping core reactions.
        Defaults to 0.333.
    evidence_scores : dict[str, Union[int, float]] or pd.Series, optional
        Additional evidence scores for reactions (e.g., from literature, proteomics).
        Higher scores favor keeping the reaction. Defaults to None (all zero).
    salvage_check_tasks : TaskContainer or str, optional
        Metabolic tasks (e.g., salvage pathways) that the final model must be
        able to perform. Can be a TaskContainer object or a path to a task file.
        Defaults to None.
    default_salv_test : bool, optional
        If True, use predefined default salvage pathway tasks (Guanine -> GMP,
        Hypoxanthine -> IMP). Defaults to False.
    func_test_tasks : TaskContainer or str, optional
        General metabolic function tasks that the final model must be able to
        perform. Defaults to None.
    required_met_ids : list[str], optional
        List of metabolite IDs that the model must be able to produce (used if
        `default_func_test` is True). Defaults to None.
    default_func_test : bool, optional
        If True and `required_met_ids` is provided, use predefined default
        functional tasks (production of each required metabolite from glucose).
        Defaults to False.

    Returns
    -------
    mCADRE_Analysis
        An object containing the results:
        - result_model (cobra.Model): The final context-specific model.
        - removed_rxn_ids (np.ndarray): IDs of removed reactions.
        - core_rxn_ids (np.ndarray): IDs of reactions initially defined as core.
        - non_expressed_rxn_ids (np.ndarray): IDs of reactions initially defined as non-expressed.
        - score_df (pd.DataFrame): DataFrame with expression, connectivity, and evidence scores.
        - salvage_test_result (TaskAnalysis or None): Results of salvage pathway tests.
        - func_test_result (TaskAnalysis or None): Results of functional tests.
        - threshold_analysis (ThresholdAnalysis): Details of thresholding used.
        - algo_efficacy (float): Efficacy score (e.g., F1) comparing the final model
          against the initial core/non-core sets.

    Raises
    ------
    RuntimeError
        If the initial model fails any of the provided functional or salvage tests.

    Notes
    -----
    Based on the algorithm described in: Wang, Y., Eddy, J. A., & Price, N. D. (2012).
    Reconstruction of genome-scale metabolic models for 126 human tissues using mCADRE.
    BMC systems biology, 6, 1-16.
    """
    threshold_dic = parse_predefined_threshold(predefined_threshold=predefined_threshold,
                                               gene_data=data.gene_data,
                                               **threshold_kws)
    th_result, exp_th, non_exp_th = threshold_dic["th_result"], threshold_dic["exp_th"], threshold_dic["non_exp_th"]

    score_df, core_rxns, non_expressed_rxns = get_score_df(model,
                                                           data,
                                                           exp_th,
                                                           non_exp_th,
                                                           protected_rxns=protected_rxns,
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
                                                 protected_rxns,
                                                 func_test_result,
                                                 salvage_test_result,
                                                 eta=eta,
                                                 consistency_test_method="FASTCC",
                                                 tolerance=tol,
                                                 rxn_scaling_coefs=rxn_scaling_coefs)
    non_core_ids = [r.id for r in model.reactions if r.id not in core_rxns]

    eff_score = measure_efficacy(kept_rxn_ids=[r.id for r in result_model.reactions],
                                 removed_rxn_ids=removed_rxn_ids,
                                 core_rxn_ids=core_rxns,
                                 non_core_rxn_ids=non_core_ids)

    result = mCADRE_Analysis(log=dict(exp_cutoff=exp_cutoff,
                                      absent_value=absent_value,
                                      absent_value_indicator=absent_value_indicator,
                                      evidence_scores=evidence_scores,
                                      tol=tol,
                                      eta=eta,))

    result.add_result(dict(result_model=result_model,
                           removed_rxn_ids=np.array(removed_rxn_ids),
                           core_rxn_ids=np.array(core_rxns),
                           non_expressed_rxn_ids=np.array(non_expressed_rxns),
                           score_df=score_df,
                           salvage_test_result=salvage_test_result,
                           func_test_result=func_test_result,
                           threshold_analysis=th_result,
                           algo_efficacy=eff_score))
    return result
