import numpy as np
import pandas as pd
from cobra.util.array import create_stoichiometric_matrix
from pipeGEM.analysis.tasks import Task, TaskContainer, TaskHandler


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
                 absent_value=0,
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


def apply_mCADRE(model,
                 data,
                 expr_th,
                 nonexpr_th,
                 exp_cutoff=0.9,
                 absent_value=0,
                 absent_value_indicator=-1e-6,
                 evidence_scores=None,
                 salvage_check_tasks=None,
                 func_test_tasks=None,
                 required_met_ids=None,
                 use_default_inp_for_func_test=False,
                 ):
    score_df, core_rxns, non_expressed_rxns = get_score_df(model,
                                                           data,
                                                           expr_th,
                                                           nonexpr_th,
                                                           exp_cutoff=exp_cutoff,
                                                           absent_value=absent_value,
                                                           absent_value_indicator=absent_value_indicator,
                                                           evidence_scores=evidence_scores)

    if required_met_ids is not None:
        if use_default_inp_for_func_test:
            tasks = _get_mCADRE_default_func_test_tasks(required_met_ids)
            func_task_handler = TaskHandler(model=model, tasks_path_or_container=tasks)