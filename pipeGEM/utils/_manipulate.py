import re
from warnings import warn
from typing import List

import numpy as np
import pandas as pd
import cobra

from pipeGEM.utils._selection import get_objective_rxn, get_not_met_exs, get_organic_exs


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


def sep_isoenzymes(model, name_format="{rxn.id}_iso_{i}"):
    for r in model.reactions:
        r.gene_reaction_rule

def sep_reversible_rxns(model):
    pass


def get_rev_arr(model):
    lbs = np.array([r.lower_bound for r in model.reactions])
    ubs = np.array([r.upper_bound for r in model.reactions])
    return (lbs < 0) & (ubs > 0)


def random_perturb(model,
                   in_place=True,
                   on_constr=True,
                   on_structure=True,
                   structure_ratio=0.8,
                   constr_ratio=0.8,
                   random_state: int = 34):
    if not in_place:
        model = model.copy()

    rng = np.random.default_rng(random_state)
    if on_structure:
        n = len(model.reactions)
        objs = get_objective_rxn(model)
        amptitude = rng.uniform(low=structure_ratio, high=1)
        exchanges = [ex.id for ex in model.exchanges]
        removed = rng.choice([r.id for r in model.reactions if r.id not in objs+exchanges],
                             size=(int(n * (1 - structure_ratio * amptitude)),), replace=False)
        model.remove_reactions(removed, remove_orphans=True)

    if on_constr:
        n = len(model.reactions)
        magnitudes = rng.uniform(low=constr_ratio, high=1, size=(n,))
        for i, rxn in enumerate(model.reactions):
            rxn.lower_bound, rxn.upper_bound = rxn.lower_bound * magnitudes[i], rxn.upper_bound * magnitudes[i]
    return model


def sheet_to_comp(model, excel_file_name, raise_err=False):
    mets = pd.read_excel(excel_file_name, sheet_name="metabolites")
    rxns = pd.read_excel(excel_file_name, sheet_name="reactions")

    exist_mets = [m.id for m in model.metabolites]
    exist_rxns = [m.id for m in model.metabolites]
    added_mets = []
    added_rxns = []
    for m in mets.iterrows():
        if raise_err:
            assert m[1]['ID'] not in exist_mets, f"The metabolite {m[1]['ID']} is already in the model."
        new_met = cobra.Metabolite(id=m[1]["ID"],
                                   formula=m[1]["formula"],
                                   name=m[1]["name"],
                                   compartment=m[1]["compartment"])
        if new_met in exist_mets:
            warn(f"The metabolite {m[1]['ID']} is already in the model.")
        else:
            added_mets.append(new_met)
    model.add_metabolites(added_mets)

    for r in rxns.iterrows():
        if raise_err:
            assert r[1]['ID'] not in exist_mets, f"The reaction {r[1]['ID']} is already in the model."
        added_rxn = cobra.Reaction(id=r[1]["ID"],
                                   name=r[1]["name"],
                                   lower_bound=r[1]["lower_bound"],
                                   upper_bound=r[1]["upper_bound"])
        GPR = str(r[1]["GPR"]) if r[1]["GPR"] is not None else ""
        added_rxn.gene_reaction_rule = GPR
        if added_rxn in exist_rxns:
            warn(f"The reaction {r[1]['ID']} is already in the model.")
        else:
            added_rxns.append(added_rxn)
            model.add_reactions([added_rxn])

            added_rxn.build_reaction_from_string(r[1]["reaction"])



def apply_medium_constraint(model,
                            mets_in_medium,
                            exclude_others: bool = True,
                            except_rxns: List[str] = None) -> list:
    # remove all carbon sources but medium composition
    if except_rxns is None:
        except_rxns = []
    if exclude_others:
        to_ko = get_not_met_exs(model, mets_in_medium, get_objective_rxn(model) + except_rxns)
    else:  # rFastCormic original function
        to_ko = get_organic_exs(model, mets_in_medium, get_objective_rxn(model) + except_rxns)
    constraint_rxns = []
    for r in to_ko:
        if r.lower_bound != 0:
            constraint_rxns.append(r.id)
            print("apply constraint to ", r.id)
        r.lower_bound = 0
    return constraint_rxns


def flip_direction(model, to_flip: List[str]):
    for rxn_id in to_flip:
        rxn = model.reactions.get_by_id(rxn_id)
        rxn.lower_bound, rxn.upper_bound = -rxn.upper_bound if rxn.upper_bound != 0 else 0, -rxn.lower_bound if rxn.lower_bound != 0 else 0
        rxn.subtract_metabolites({
            met: 2 * c for met, c in rxn.metabolites.items()
        })


def remove_drug_rxns_from_human_gem(mod):
    drugs = ['pravastatin', 'Gliclazide', 'atorvastatin', 'fluvastatin', 'fluvastain','fluvstatin',
             'simvastatin', 'cyclosporine',
             'acetaminophen','cerivastatin','Tacrolimus', 'ibuprofen', 'lovastatin','Losartan','nifedipine',
             'pitavastatin','rosuvastatin','Torasemide','Midazolam']
    drugs = [d.lower() for d in drugs]

    m_list = []
    for m in mod.metabolites:
        if m.name.lower() in drugs:
            m_list.append(m)

    print(f"Removing {len(m_list)} mets from the model..")
    mod.remove_metabolites(m_list, destructive=True)

    isolated_mets = []
    for m in mod.metabolites:
        if len(m.reactions) == 0:
            isolated_mets.append(m)

    mod.remove_metabolites(isolated_mets, True)


def remove_AA_triplet_rxns_from_human_gem(mod):
    aa_caps = ['Alanine', 'Arginine', 'Asparagine', 'Aspartate', 'Cysteine',
               'Glutamine', 'Glutamate', 'Glycine', 'Histidine',
               'Isoleucine', 'Leucine', 'Lysine', 'Methionine', 'Metheonine',
               'Phenylalanine', 'Proline', 'Serine', 'Threonine',
               'Tryptophan', 'Tyrosine', 'Valine']
    aa_prefs = ['Alanyl', 'Alaninyl', 'Alanine',
                'Arginyl', 'Asparaginyl', 'Aspartyl',
                'Cystyl', 'Cystinyl', 'Cysteinyl',
                'Glutaminyl', 'Glutamyl', 'Glutamatsyl',
                'Glycyl', 'Histidyl', 'Histidinyl', 'Isoleucyl',
                'Isolecyl', 'Leucyl', 'Lysyl', 'Lysine', 'Methionyl', 'Methioninyl', 'Phenylalanyl',
                'Phenylalanine', 'Phenylalaninyl', 'Prolyl', 'Seryl', 'Threonyl', 'Tryptophanyl',
                'Tyrosyl', 'Tyrosinyl', 'Valyl']
    aa_caps = [i.lower() for i in aa_caps]
    aa_prefs = [i.lower() for i in aa_prefs]
    single_trip_name = 'argtyrval'
    to_remove = []

    for m in mod.metabolites:
        if m.name.lower() == single_trip_name:
            print(m.name, " is found")
            to_remove.append(m)
        split_name = m.name.split("-")
        if len(split_name) <= 1:
            continue
        if all([n.lower() in aa_prefs for n in split_name[:-1]]) and (split_name[-1].lower() in aa_caps):
            to_remove.append(m)
    print(f"Removing {len(to_remove)} mets from the model..")
    mod.remove_metabolites(to_remove, destructive=True)

    isolated_mets = []
    for m in mod.metabolites:
        if len(m.reactions) == 0:
            isolated_mets.append(m)

    mod.remove_metabolites(isolated_mets, True)