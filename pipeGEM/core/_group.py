from typing import List

from pipeGEM.core._base import GEMComposite
from pipeGEM.core._model import Model


class Group(GEMComposite):
    _is_leaf = False

    def __init__(self,
                 group,
                 name_tag: str = None,
                 data=None):
        super().__init__(name_tag=name_tag)
        self.data = data
        self._lvl = 0
        self._group: List[GEMComposite] = self._form_group(group)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Group [{self._name_tag}]"

    def __len__(self):
        return len(self._group)

    def __iter__(self):
        index = 0
        while index < len(self._group):
            yield self._group[index]
            index += 1

    def tget(self, tag):
        if isinstance(tag, str):
            selected = [g for g in self._group if g.name_tag == tag]
        elif isinstance(tag, list):
            if len(tag) > 1:
                selected = [g.tget(tag[1:]) if not g.is_leaf else g for g in self._group if g.name_tag == tag[0]]
            else:
                selected = self.tget(tag[0])
        else:
            raise ValueError
        return selected

    def iget(self, index):
        if isinstance(index, int):
            selected = self._group[index]
        elif isinstance(index, list):
            if len(index) > 1:
                selected = [g.tget(index[1:]) if not g.is_leaf else g for g in self._group[index[0]]]
            else:
                selected = self.iget(index[0])
        else:
            raise ValueError
        return selected

    @property
    def members(self):
        return "\n".join([str(g) for g in self._group])

    def _form_group(self, group_dict) -> list:
        group_lis = []
        max_g = 0
        for name, comp in group_dict.items():
            if isinstance(comp, dict):
                g = Group(group=comp, name_tag=name, data=self.data)
                max_g = max(max_g, g._lvl)
                group_lis.append(g)
            else:
                group_lis.append(Model(model=comp, name_tag=name, data=self.data))
                max_g = max(max_g, 0)
        self._lvl = max_g + 1
        return group_lis

    def _traverse_util(self, comp: GEMComposite, suffix_row, max_lvl):
        assert max_lvl >= len(suffix_row)
        if comp.is_leaf:
            return suffix_row + ["-" for _ in range(max_lvl - len(suffix_row))] + [len(comp.reactions), len(comp.metabolites), len(comp.genes)]
        return [self._traverse_util(c, suffix_row + [c.name_tag], max_lvl) for c in comp]

    def _traverse(self, tag=None, index=None):
        assert (tag is not None) ^ (index is not None)
        if tag is not None:
            comps = self.tget(tag)
        else:
            comps = self.iget(index)
        max_lvl = max([c._lvl for c in comps])
        data = []
        for c in comps:
            data += self._traverse_util(c, [], max_lvl=max_lvl)
        return data
