from typing import List

import cobra

from pipeGEM.utils import get_objective_rxn


def get_organic_exs(model,
                    except_mets: List[str],
                    except_rxns: List[str]) -> List[cobra.Reaction]:
    ex_rxns = model.exchanges + model.sinks + model.demands
    return [r
            for r in ex_rxns
            if r.id not in except_rxns and
               all([m.id not in except_mets for m in r.metabolites]) and
               any(["C" in m.formula for m in r.metabolites])]


def find_not_met_exs(model,
                     mets: List[str],
                     except_rxns) -> List[cobra.Reaction]:
    ex_rxns = model.exchanges + model.sinks + model.demands
    to_ko_ids = list(set([r.id for r in ex_rxns]) - set([r.id
                                                         for r in model.exchanges
                                                         if any([m.id in mets for m in r.metabolites])] + except_rxns))
    return [r for r in ex_rxns if r.id in to_ko_ids]


def apply_medium_constraint(model,
                            mets_in_medium,
                            exclude_others: bool = True,
                            except_rxns: List[str] = None) -> list:
    # remove all carbon sources but medium composition
    if except_rxns is None:
        except_rxns = []
    if exclude_others:
        to_ko = find_not_met_exs(model, mets_in_medium, get_objective_rxn(model) + except_rxns)
    else:  # rFastCormic original function
        to_ko = get_organic_exs(model, mets_in_medium, get_objective_rxn(model) + except_rxns)
    constraint_rxns = []
    for r in to_ko:
        if r.lower_bound != 0:
            constraint_rxns.append(r.id)
            print("apply constraint to ", r.id)
        r.lower_bound = 0
    return constraint_rxns


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