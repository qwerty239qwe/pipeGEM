import re
from typing import List

import pandas as pd
import cobra


def _join_str(main_list, glue_pos, glue_str='and'):
    """
    Helper function of unify_grr
    """
    new_str_list = list()
    ptr = 0
    for glue_lis in glue_pos:
        if glue_lis[0] != ptr:
            # dump
            new_str_list.extend(main_list[ptr: glue_lis[0]])
        new_str_list.append('({})'.format(f' {glue_str} '.join(main_list[glue_lis[0]: glue_lis[-1] + 2])))
        ptr = glue_lis[-1] + 2
    return new_str_list


def unify_grr(grr_str):
    if len(grr_str) == 0 or grr_str == r'()':
        return ''

    pattern = re.compile(r'\d+\.\d')
    pattern_andor = re.compile(r'or|and')
    gene_list = re.findall(pattern, grr_str)
    gene_list = [str(int(float(g))) for g in gene_list]
    conn_list = re.findall(pattern_andor, grr_str)
    if len(gene_list) - len(conn_list) != 1:
        print('This string is problematic, please check its pattern')
        return grr_str

    and_ind_list, and_ind_lol = [], []
    for i, c in enumerate(conn_list):
        if c == 'and':
            and_ind_list.append(i)
        else:
            if len(and_ind_list) != 0:
                and_ind_lol.append(and_ind_list)
            and_ind_list = []
    if len(and_ind_list) != 0:
        and_ind_lol.append(and_ind_list)
    if len(and_ind_lol) != 0:
        gene_list = _join_str(gene_list, and_ind_lol, 'and')
    return ' or '.join(set(gene_list))  # unique relationships


def make_DM_met_rxn(mod, met_id, name, lb, ub):
    dummy_rxn = cobra.Reaction(name, lower_bound=lb, upper_bound=ub)
    dummy_rxn.add_metabolites({
        mod.metabolites.get_by_id(met_id): -1
    })
    return dummy_rxn


def remove_constr(mod, rxn_ids, make_rev=False, upper_bound=1000, lower_bound=-1000):
    for rxn_id in rxn_ids:
        rxn = mod.reactions.get_by_id(rxn_id)
        is_rev = bool(rxn.lower_bound * rxn.upper_bound)
        if is_rev or make_rev:
            rxn.lower_bound, rxn.upper_bound = -upper_bound, upper_bound
        else:
            rxn.lower_bound = -upper_bound if rxn.lower_bound != 0 else lower_bound
            rxn.upper_bound = upper_bound if rxn.upper_bound != 0 else lower_bound


def make_irrev_rxn(mod,
                   rxn_id,
                   add_inplace=True,
                   ignore_irrev=False,
                   forward_prefix="_F_",
                   backward_prefix="_R_") -> List[cobra.Reaction]:
    rxn = mod.reactions.get_by_id(rxn_id)

    new_rxns = []
    if not (ignore_irrev and rxn.upper_bound == 0):
        forward_rxn = cobra.Reaction(f"{forward_prefix}{rxn.id}", upper_bound=rxn.upper_bound, lower_bound=0)
        forward_rxn.add_metabolites({
            met: c for met, c in rxn.metabolites.items()
        })
        new_rxns.append(forward_rxn)
    if not (ignore_irrev and rxn.lower_bound == 0):
        reversed_rxn = cobra.Reaction(f"{backward_prefix}{rxn.id}", upper_bound=-rxn.lower_bound, lower_bound=0)
        reversed_rxn.subtract_metabolites({
            met: c for met, c in rxn.metabolites.items()
        })
        new_rxns.append(reversed_rxn)
    if add_inplace:
        mod.add_reactions(new_rxns)
    return new_rxns


def merge_irrevs_in_df(sol_df: pd.DataFrame,
                       forward_prefix="_F_",
                       backward_prefix="_R_"
                       ):
    f_rxns = sol_df.index.str.match(f"^{forward_prefix}")
    b_rxns = sol_df.index.str.match(f"^{backward_prefix}")
    f_df = sol_df[f_rxns]
    f_df.index = f_df.index.to_series().str.partition(forward_prefix).iloc[:, 2]
    f_df.columns = ["forward", "reduced_costs"]
    b_df = sol_df[b_rxns]
    b_df.index = b_df.index.to_series().str.partition(backward_prefix).iloc[:, 2]
    b_df.columns = ["backward", "reduced_costs"]
    sol_df = sol_df[(~f_rxns) & (~b_rxns)]
    sol_df = sol_df.merge(f_df, how="outer",
                         left_index=True, right_index=True).merge(b_df, how="outer",
                                                                  left_index=True, right_index=True).fillna(0)
    sol_df["fluxes"] = sol_df["fluxes"] + sol_df["forward"] - sol_df["backward"]
    return sol_df