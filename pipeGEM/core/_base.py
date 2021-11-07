

class GEMComposite:
    _is_leaf = None

    def __init__(self,
                 name_tag):
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
    def reaction_ids(self):
        raise NotImplementedError()

    @property
    def metabolite_ids(self):
        raise NotImplementedError()

    @property
    def gene_ids(self):
        raise NotImplementedError()

    @property
    def order(self):
        return self._order