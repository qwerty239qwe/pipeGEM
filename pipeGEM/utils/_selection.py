from typing import List, Any
from warnings import warn
from functools import reduce
import re

import cobra

__all__ = ("get_objective_rxn", "get_subsystems", "get_rxns_in_subsystem", "get_rxn_set",
           "get_organic_exs", "get_not_met_exs",
           "get_genes_in_subsystem", "select_rxns_from_model")

import numpy as np


def get_objective_rxn(model, attr="id") -> List[Any]:
    return [getattr(r, attr) if attr else r
            for r in model.reactions if r.objective_coefficient != 0]


def get_subsystems(model):
    all_subs = []
    for rxn in model.reactions:
        if rxn.subsystem is None:
            continue
        if isinstance(rxn.subsystem, str):
            all_subs.append(rxn.subsystem)
        else:
            all_subs.extend(list(rxn.subsystem))

    return list(set(all_subs))


def get_rxns_in_subsystem(model, subsystem, attr="id") -> list:
    rxn_prop = []
    for r in model.reactions:
        if isinstance(r.subsystem, str):
            if (r.subsystem in subsystem) if isinstance(subsystem, list) else re.match(subsystem, r.subsystem):
                rxn_prop.append(getattr(r, attr) if attr != "" else r)
        elif isinstance(r.subsystem, list) or isinstance(r.subsystem, np.ndarray):
            if (isinstance(subsystem, list) and len(set(r.subsystem) & set(subsystem)) > 0) or \
               any([re.match(subsystem, s) for s in r.subsystem]):
                rxn_prop.append(getattr(r, attr) if attr != "" else r)

    return rxn_prop


def get_genes_in_subsystem(model, subsystem):
    r_in_sub = get_rxns_in_subsystem(model, subsystem, "")
    return list(set([g.id for r in r_in_sub for g in r.genes]))


def select_rxn_from_model(named_model, query_id, return_id: bool = True):
    assert isinstance(query_id, str), f"query_id should be a str but a {type(query_id)} is passed"
    if query_id in get_subsystems(named_model):
        return [rxn.id if return_id else rxn for rxn in named_model.reactions if rxn.subsystem == query_id]
    for rxn in named_model.reactions:
        if rxn.id == query_id:
            return [rxn.id if return_id else rxn]
    warn(f"{query_id} is not in the {str(named_model)}")
    return []


def select_rxns_from_model(named_model, query_id, return_id: bool = True) -> list:
    if isinstance(query_id, str):
        return select_rxn_from_model(named_model, query_id, return_id)
    elif isinstance(query_id, list):
        return list(reduce(set.union, [set(select_rxn_from_model(named_model, qid, return_id)) for qid in query_id]))

    raise ValueError(f"query_id should be a str or a list but a {type(query_id)} is passed")


def get_rxn_set(model, cond: str = "", dtype=set):
    if cond == "irreversible":
        return dtype([rxn.id for rxn in model.reactions
                    if not rxn.reversibility])
    if cond == "not_expressed":
        return dtype([rxn.id for rxn in model.reactions
                    if (rxn.lower_bound == 0 and rxn.upper_bound == 0) or not rxn.functional])
    if cond == "backward":  # ub <= 0 and lb < 0
        return dtype([rxn.id for rxn in model.reactions
                    if rxn.upper_bound <= 0 and rxn.lower_bound < 0])
    if cond == "forward":  # ub > 0 and lb >= 0
        return dtype([rxn.id for rxn in model.reactions
                    if rxn.upper_bound > 0 and rxn.lower_bound >= 0])
    if cond == "reversed_single_met":  # S = -1
        return dtype([rxn.id for rxn in model.reactions
                    if len(rxn.metabolites) == 1 and all([c < 0 for m, c in rxn.metabolites.items()])])
    if cond == "forward_single_met":  # S = 1
        return dtype([rxn.id for rxn in model.reactions
                    if len(rxn.metabolites) == 1 and all([c > 0 for m, c in rxn.metabolites.items()])])

    return dtype([rxn.id for rxn in model.reactions])


def get_organic_exs(model,
                    except_mets: List[str],
                    except_rxns: List[str]) -> List[cobra.Reaction]:
    ex_rxns = model.exchanges + model.sinks + model.demands
    organic_ex = []
    for r in ex_rxns:
        if r.id in except_rxns:
            continue
        if any([m.id in except_mets for m in r.metabolites]):
            continue
        if any(["C" in m.formula and "H" in m.formula for m in r.metabolites]) or \
                any(["C" in m.formula and "R" in m.formula for m in r.metabolites]) or \
                any(["C" in m.formula and "X" in m.formula for m in r.metabolites]) or \
                any(["X" in m.formula and "H" in m.formula for m in r.metabolites]) or \
                any(["X" in m.formula and "R" in m.formula for m in r.metabolites]):
            organic_ex.append(r)
    return organic_ex


def get_not_met_exs(model,
                    mets: List[str],
                    except_rxns) -> List[cobra.Reaction]:
    ex_rxns = model.exchanges + model.sinks + model.demands
    to_ko_ids = list(set([r.id for r in ex_rxns]) - set([r.id
                                                         for r in model.exchanges
                                                         if any([m.id in mets for m in r.metabolites])] + except_rxns))
    return [r for r in ex_rxns if r.id in to_ko_ids]