import cobra


def _assign_grp_index(grp_indexes, new_grp_id, rxn1, rxn2):
    cur_grp_id = max(grp_indexes.values())
    if cur_grp_id == 0:
        cur_grp_id = new_grp_id
        new_grp_id += 1
    if grp_indexes[rxn1.id] != cur_grp_id:
        if grp_indexes[rxn1.id] == 0:
            grp_indexes[rxn1.id] = cur_grp_id
        else:
            grp_indexes = {k: cur_grp_id if v == grp_indexes[rxn1.id] else v
                           for k, v in grp_indexes.items()}

    if grp_indexes[rxn2.id] != cur_grp_id:
        if grp_indexes[rxn2.id] == 0:
            grp_indexes[rxn2.id] = cur_grp_id
        else:
            grp_indexes = {k: cur_grp_id if v == grp_indexes[rxn2.id] else v
                           for k, v in grp_indexes.items()}
    return grp_indexes, new_grp_id


def _iter_merge_rxns(model, not_merged_rxns, grp_indexes, new_grp_id):
    mets_to_merge = [m for m in model.metabolites if len(m.reactions) == 2]
    not_merged_rxns = set(not_merged_rxns)
    to_continue_flag = False
    for m in mets_to_merge:
        if len(set(not_merged_rxns) & set([r.id for r in m.reactions])) != 0 or len([r.id for r in m.reactions]) != 2:
            continue
        pos = [r.id for r in m.reactions if r.reversibility or (r.metabolites[m] > 0 and r.upper_bound > 0)]
        neg = [r.id for r in m.reactions if r.reversibility or (r.metabolites[m] < 0 and r.lower_bound < 0)]

        if len(pos) < 1 or len(neg) < 1:
            continue
        tb_merged = m.reactions
        merged_rxn = cobra.Reaction(f"merged_rxn_of_{tb_merged[0].id}_and_{tb_merged[1].id}")
        s_ratio = -tb_merged[0].metabolites[m] / tb_merged[1].metabolites[m]
        to_rev = s_ratio > 0
        merged_rxn.add_metabolites(tb_merged[0].metabolites)
        merged_rxn.add_metabolites({k: v * s_ratio for k, v in tb_merged[1].metabolites.items()})
        if m in merged_rxn.metabolites:
            merged_rxn.subtract_metabolites({m: merged_rxn.metabolites[m]})
        merged_rxn.lower_bound = max(tb_merged[0].lower_bound,
                                     (-tb_merged[1].upper_bound if to_rev else tb_merged[1].lower_bound) / s_ratio)

        merged_rxn.upper_bound = min(tb_merged[0].upper_bound,
                                     (-tb_merged[1].lower_bound if to_rev else tb_merged[1].upper_bound) / s_ratio)
        grp_indexes = _assign_grp_index(grp_indexes, new_grp_id=new_grp_id, rxn1=tb_merged[0], rxn2=tb_merged[1])
        model.add_reactions([merged_rxn])
        model.remove_reactions([tb_merged[0], tb_merged[1]], remove_orphans=True)
        to_continue_flag = True
    return to_continue_flag, new_grp_id, grp_indexes


def merge_linear_rxns(model, not_merged_rxns):
    reduced_model = model.copy()
    new_grp_id = 1
    to_continue_flag = True

    while to_continue_flag:
        grp_indexes = {r.id: 0 for r in reduced_model.reactions}
        to_continue_flag, new_grp_id, grp_indexes = _iter_merge_rxns(reduced_model, not_merged_rxns,
                                                                     grp_indexes, new_grp_id)

    return reduced_model, grp_indexes

