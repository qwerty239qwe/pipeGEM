import cobra
import itertools


class MergedReaction(cobra.Reaction):
    def __init__(self, id, name=None, lower_bound=0, upper_bound=1000):
        super().__init__(id=id, name=name, lower_bound=lower_bound, upper_bound=upper_bound)
        self._merged_rxns = []

    @property
    def merged_rxns(self) -> list:
        return self._merged_rxns

    def merge_two_rxns(self, merged_rxns, shared_met):
        self._merged_rxns = [[r] if not hasattr(r, "merged_rxns") else r.merged_rxns for r in merged_rxns]
        self._merged_rxns = itertools.chain(*self._merged_rxns)
        s_ratio = -merged_rxns[0].metabolites[shared_met] / merged_rxns[1].metabolites[shared_met]
        to_rev = s_ratio > 0
        self.add_metabolites(merged_rxns[0].metabolites)
        self.add_metabolites({k: v * s_ratio for k, v in merged_rxns[1].metabolites.items()})
        if shared_met in self.metabolites:
            self.subtract_metabolites({shared_met: self.metabolites[shared_met]})
        self.lower_bound = max(merged_rxns[0].lower_bound,
                               (-merged_rxns[1].upper_bound if to_rev else merged_rxns[1].lower_bound) / s_ratio)

        self.upper_bound = min(merged_rxns[0].upper_bound,
                               (-merged_rxns[1].lower_bound if to_rev else merged_rxns[1].upper_bound) / s_ratio)

    @property
    def genes(self):
        return frozenset().union(*[r.genes for r in self._merged_rxns])

    @property
    def gene_reaction_rule(self):
        return " plus ".join([f"({r.gene_reaction_rule})" for r in self._merged_rxns])


# def _assign_grp_index(grp_indexes, new_grp_id, rxn1, rxn2):
#     cur_grp_id = max(grp_indexes.values())
#     if cur_grp_id == 0:
#         cur_grp_id = new_grp_id
#         new_grp_id += 1
#     if grp_indexes[rxn1.id] != cur_grp_id:
#         if grp_indexes[rxn1.id] == 0:
#             grp_indexes[rxn1.id] = cur_grp_id
#         else:
#             grp_indexes = {k: cur_grp_id if v == grp_indexes[rxn1.id] else v
#                            for k, v in grp_indexes.items()}
#
#     if grp_indexes[rxn2.id] != cur_grp_id:
#         if grp_indexes[rxn2.id] == 0:
#             grp_indexes[rxn2.id] = cur_grp_id
#         else:
#             grp_indexes = {k: cur_grp_id if v == grp_indexes[rxn2.id] else v
#                            for k, v in grp_indexes.items()}
#     return grp_indexes, new_grp_id


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
        merged_rxn = MergedReaction(f"{tb_merged[0].id}_and_{tb_merged[1].id}")
        merged_rxn.merge_two_rxns(tb_merged, shared_met=m)
        # grp_indexes = _assign_grp_index(grp_indexes, new_grp_id=new_grp_id, rxn1=tb_merged[0], rxn2=tb_merged[1])
        model.add_reactions([merged_rxn])
        model.remove_reactions([tb_merged[0], tb_merged[1]], remove_orphans=True)
        to_continue_flag = True
    return to_continue_flag


def merge_linear_rxns(model, not_merged_rxns):
    reduced_model = model.copy()
    new_grp_id = 14
    to_continue_flag = True

    while to_continue_flag:
        grp_indexes = {r.id: 0 for r in reduced_model.reactions}
        to_continue_flag, new_grp_id, grp_indexes = _iter_merge_rxns(reduced_model, not_merged_rxns,
                                                                     grp_indexes, new_grp_id)

    return reduced_model

