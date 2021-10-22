from typing import List, Any


__all__ = ("get_objective_rxn", "get_subsystems", "get_rxns_in_subsys", "get_genes_in_subsys")


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