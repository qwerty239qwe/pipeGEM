from typing import List

import cobra

from pipeGEM.utils import get_objective_rxn, get_organic_exs, get_not_met_exs


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


def flip_direction(model, to_flip: List[str]):
    for rxn_id in to_flip:
        rxn = model.reactions.get_by_id(rxn_id)
        rxn.lower_bound, rxn.upper_bound = -rxn.upper_bound if rxn.upper_bound != 0 else 0, -rxn.lower_bound if rxn.lower_bound != 0 else 0
        rxn.subtract_metabolites({
            met: 2 * c for met, c in rxn.metabolites.items()
        })