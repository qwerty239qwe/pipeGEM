import cobra


def add_fake_mets_for_core_mets(model, met_ids, added_comp="c"):
    # create fake met for essential mets
    mets_in_mod = [m.id for m in model.metabolites]
    new_mets = []
    for met_id in met_ids:
        if met_id in mets_in_mod:
            pm: cobra.Metabolite = model.metabolites.get_by_id(met_id)
            new_mets.append(cobra.Metabolite(id = "fake_PM_"+met_id,
                                             formula=pm.formula,
                                             name="fake_PM_"+pm.name,
                                             charge=pm.charge,
                                             compartment=added_comp
                                             ))
    return new_mets


def add_fake_mets_for_non_core_rxns(model, non_cores):
    rxns_in_mod = [r.id for r in model.reactions]
    assert all([nc in rxns_in_mod for nc in non_cores])
    new_mets, new_rxns = [], []
    for r in non_cores:
        fm = cobra.Metabolite(id = f"fake_met_for_{r}")
        fr = cobra.Reaction(id = f"fake_rxn_for_{r}", lower_bound=0, upper_bound=1)
        model.reactions.get_by_id(r).add_metabolites({fm: 1})
        new_mets.append(fm)
        new_rxns.append(fr)
    model.add_reactions(new_rxns)
    return new_mets, new_rxns


def add_sinks_for_mets(model, real_met_ids):
    mets_in_mod = [m.id for m in model.metabolites]
    new_rxns = []
    for m in real_met_ids:
        fr = cobra.Reaction(id = f"sink_for_{m}", lower_bound=0, upper_bound=1)
        new_rxns.append(fr)
    model.add_reactions(new_rxns)
    for mid, r in zip(real_met_ids, new_rxns):
        r.add_metabolites({model.metabolites.get_by_id(mid): -1})
    return new_rxns


def add_constr_for_rev_rxns(model, forward_rxn_ids, reverse_rxn_ids):
    # the forward_rxn_ids and the reverse_rxn_ids should be paired
    for fr, rr in zip(forward_rxn_ids, reverse_rxn_ids):
        im_f = cobra.Metabolite(f"indicator_met_for_{fr}")
        model.reactions.get_by_id(fr).add_metabolites({im_f: 1})
        im_r = cobra.Metabolite(f"indicator_met_for_{rr}")
        model.reactions.get_by_id(rr).add_metabolites({im_r: 1})
        ir = cobra.Reaction(f"indicator_rxn_for_{fr}")


def build_INIT_problem(model, ):
    pass

# ================== ^ old algo ^ ==================


