from typing import List, Any
from warnings import warn
from functools import reduce

import cobra

__all__ = ("get_objective_rxn", "get_subsystems", "get_rxns_in_subsys", "get_rxn_set",
           "get_genes_in_subsys", "select_rxns_from_model")


def get_objective_rxn(model, attr="id") -> List[Any]:
    return [getattr(r, attr) if attr else r
            for r in model.reactions if r.objective_coefficient != 0]


def get_subsystems(model):
    return list(set([rxn.subsystem for rxn in model.reactions]))


def get_rxns_in_subsys(model, subsys_id, dtype="id", match_all=True):
    return [getattr(r, dtype) if dtype != "" else r for r in model.reactions
            if (r.subsystem == subsys_id if match_all else subsys_id in r.subsystem)]


def get_genes_in_subsys(model, subsys_id):
    r_in_sub = get_rxns_in_subsys(model, subsys_id, "")
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


def get_rxn_set(model, cond: str = ""):
    if cond == "irreversible":
        return set([rxn.id for rxn in model.reactions
                    if not rxn.reversibility])
    if cond == "not_expressed":
        return set([rxn.id for rxn in model.reactions
                    if (rxn.lower_bound == 0 and rxn.upper_bound == 0) or not rxn.functional])
    if cond == "backward":  # ub <= 0 and lb < 0
        return set([rxn.id for rxn in model.reactions
                    if rxn.upper_bound <= 0 and rxn.lower_bound < 0])
    if cond == "forward":  # ub > 0 and lb >= 0
        return set([rxn.id for rxn in model.reactions
                    if rxn.upper_bound > 0 and rxn.lower_bound >= 0])
    if cond == "reversed_single_met":  # S = -1
        return set([rxn.id for rxn in model.reactions
                    if len(rxn.metabolites) == 1 and all([c < 0 for m, c in rxn.metabolites.items()])])
    if cond == "forward_single_met":  # S = 1
        return set([rxn.id for rxn in model.reactions
                    if len(rxn.metabolites) == 1 and all([c > 0 for m, c in rxn.metabolites.items()])])

    return set([rxn.id for rxn in model.reactions])


def get_organic_exs(model,
                    except_mets: List[str],
                    except_rxns: List[str]) -> List[cobra.Reaction]:
    ex_rxns = model.exchanges + model.sinks + model.demands
    return [r
            for r in ex_rxns
            if r.id not in except_rxns and
               all([m.id not in except_mets for m in r.metabolites]) and
               any(["C" in m.formula for m in r.metabolites])]


def get_not_met_exs(model,
                     mets: List[str],
                     except_rxns) -> List[cobra.Reaction]:
    ex_rxns = model.exchanges + model.sinks + model.demands
    to_ko_ids = list(set([r.id for r in ex_rxns]) - set([r.id
                                                         for r in model.exchanges
                                                         if any([m.id in mets for m in r.metabolites])] + except_rxns))
    return [r for r in ex_rxns if r.id in to_ko_ids]