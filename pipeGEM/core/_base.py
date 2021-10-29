

class GEMComposite:
    _is_leaf = None

    def __init__(self, name_tag):
        self._name_tag = name_tag

    def __str__(self):
        return ""

    @property
    def is_leaf(self):
        return self._is_leaf

    @property
    def name_tag(self):
        return self._name_tag

    def tget(self, tag):
        raise NotImplementedError()