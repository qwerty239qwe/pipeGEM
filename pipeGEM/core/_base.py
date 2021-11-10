import pandas as pd


class GEMComposite:
    _is_leaf = None

    def __init__(self,
                 name_tag):
        self._lvl = 0
        self._name_tag = name_tag
        self._order = -1

    def __str__(self):
        return ""

    @property
    def is_leaf(self):
        return self._is_leaf

    @property
    def name_tag(self):
        return self._name_tag

    @property
    def n_rxns(self):
        return len(self.reaction_ids)

    @property
    def n_mets(self):
        return len(self.metabolite_ids)

    @property
    def n_genes(self):
        return len(self.gene_ids)

    @property
    def reaction_ids(self):
        raise NotImplementedError()

    @property
    def metabolite_ids(self):
        raise NotImplementedError()

    @property
    def gene_ids(self):
        raise NotImplementedError()

    @property
    def size(self):
        raise NotImplementedError()

    @property
    def order(self):
        return self._order

    def get_flux(self, as_dict=False, **kwargs):
        raise NotImplementedError()

    def do_analysis(self, **kwargs) -> None:
        raise NotImplementedError()

    @property
    def tree_level(self):
        return self._lvl