import numpy as np
import pandas as pd
from cobra.util.array import create_stoichiometric_matrix


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


def apply_mCADRE(model,
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
        evidence_scores = {r.id : 0 for r in model.reactions}
    elif isinstance(evidence_scores, dict):
        es_not_included = {r.id: 0 for r in model.reactions if r.id not in evidence_scores}
        evidence_scores = {**evidence_scores, **es_not_included}
    score_df = pd.DataFrame({"expression": expr_scores,
                             "connectivity": conn_scores,
                             "evidence": evidence_scores}).sort_values(["expression", "connectivity", "evidence"])
