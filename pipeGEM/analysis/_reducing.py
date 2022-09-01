import cobra
import itertools


class MergedReaction(cobra.Reaction):
    def __init__(self, id, name=None, lower_bound=0, upper_bound=1000):
        super().__init__(id=id, name=name, lower_bound=lower_bound, upper_bound=upper_bound)
        self._merged_rxns = {}
        self.s_ratio = 0

    @property
    def merged_rxns(self) -> dict:
        return self._merged_rxns

    def merge_two_rxns(self, merged_rxns, shared_met):

        self.s_ratio = -merged_rxns[0].metabolites[shared_met] / merged_rxns[1].metabolites[shared_met]
        to_rev = self.s_ratio < 0
        self.add_metabolites(merged_rxns[0].metabolites)
        self.add_metabolites({k: v * self.s_ratio for k, v in merged_rxns[1].metabolites.items()})
        if shared_met in self.metabolites:
            assert abs(self.metabolites[shared_met]) < 1e-4
            self.subtract_metabolites({shared_met: self.metabolites[shared_met]})
        self.lower_bound = max(merged_rxns[0].lower_bound,
                               (merged_rxns[1].upper_bound if to_rev else merged_rxns[1].lower_bound) / self.s_ratio)

        self.upper_bound = min(merged_rxns[0].upper_bound,
                               (merged_rxns[1].lower_bound if to_rev else merged_rxns[1].upper_bound) / self.s_ratio)
        mg_r_c_0 = {k: v for k, v in merged_rxns[0].merged_rxns.items()} \
            if hasattr(merged_rxns[0], "merged_rxns") else {merged_rxns[0]: 1}
        mg_r_c_1 = {k: v * self.s_ratio for k, v in merged_rxns[1].merged_rxns.items()} \
            if hasattr(merged_rxns[1], "merged_rxns") else {merged_rxns[1]: self.s_ratio}
        self._merged_rxns.update({**mg_r_c_0, **mg_r_c_1})

    @property
    def genes(self):
        return frozenset().union(*[r.genes for r in self._merged_rxns])

    @property
    def gene_reaction_rule(self):
        return " plus ".join([f"({r.gene_reaction_rule})" for r in self._merged_rxns if len(r.genes) != 0])


def _iter_merge_rxns(model, not_merged_rxns):
    mets_to_merge = [m for m in model.metabolites if len(m.reactions) == 2]
    not_merged_rxns = set(not_merged_rxns)
    to_continue_flag = False
    for i, m in enumerate(mets_to_merge):
        if (i+1) % 100 == 0 or i+1 == len(mets_to_merge):
            print(f"merging .. {i} / {len(mets_to_merge)}")
        if len(set(not_merged_rxns) & set([r.id for r in m.reactions])) != 0 or len([r.id for r in m.reactions]) != 2:
            continue
        pos = [r.id for r in m.reactions if r.reversibility or (r.metabolites[m] > 0 and r.upper_bound > 0)]
        neg = [r.id for r in m.reactions if r.reversibility or (r.metabolites[m] < 0 and r.lower_bound < 0)]

        if len(pos) < 1 or len(neg) < 1:
            continue
        tb_merged = list(m.reactions)
        merged_rxn = MergedReaction(f"MR_{m.id}", name=f"merged_rxn_{m.id}")
        merged_rxn.merge_two_rxns(tb_merged, shared_met=m)
        if len(merged_rxn.metabolites) != 0:
            model.add_reactions([merged_rxn])
        model.update_merged_rxn(merged_rxn)  # we need to do this before removing individual reactions
        model.remove_reactions([tb_merged[0], tb_merged[1]], remove_orphans=True)
        to_continue_flag = True
    return to_continue_flag


def merge_linear_rxns(model, not_merged_rxns):
    reduced_model = model.copy()
    to_continue_flag = True

    while to_continue_flag:
        to_continue_flag = _iter_merge_rxns(reduced_model, not_merged_rxns)

    return reduced_model

