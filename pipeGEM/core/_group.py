from typing import List, Union
from functools import reduce
import itertools

import numpy as np
import pandas as pd

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

    @property
    def reaction_ids(self):
        return list(reduce(set.union, [set(g.reaction_ids) for g in self._group]))

    @property
    def metabolite_ids(self):
        return list(reduce(set.union, [set(g.metabolite_ids) for g in self._group]))

    @property
    def gene_ids(self):
        return list(reduce(set.union, [set(g.gene_ids) for g in self._group]))

    def tget(self, tag):
        if isinstance(tag, str):
            if tag == self.name_tag:
                selected = [self]
            else:
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

    def _cal_jaccard_index(self, model_a_label, model_b_label, components='all'):
        if components == 'all':
            components = ['genes', 'reactions', 'metabolites']
        union_components = {'genes': set(self._group[model_a_label].gene_ids) |
                                     set(self._group[model_b_label].gene_ids),
                            'reactions': set(self._group[model_a_label].reaction_ids) |
                                         set(self._group[model_b_label].reaction_ids),
                            'metabolites': set(self._group[model_a_label].metabolite_ids) |
                                           set(self._group[model_b_label].metabolite_ids)}
        intersect_components = {'genes': set(self._group[model_a_label].gene_ids) &
                                         set(self._group[model_b_label].gene_ids),
                                'reactions': set(self._group[model_a_label].reaction_ids) &
                                             set(self._group[model_b_label].reaction_ids),
                                'metabolites': set(self._group[model_a_label].metabolite_ids) &
                                               set(self._group[model_b_label].metabolite_ids)}

        return sum([len(intersect_components[c]) for c in components]) / \
               sum([len(union_components[c]) for c in components])

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

    def _traverse_util(self, comp: GEMComposite, suffix_row, max_lvl, features):
        assert max_lvl >= len(suffix_row)
        if comp.is_leaf:
            return suffix_row + ["-" for _ in range(max_lvl - len(suffix_row))] + [len(comp.reactions), len(comp.metabolites), len(comp.genes)]
        res = []
        for c in comp:
            r = self._traverse_util(c, suffix_row + [c.name_tag], max_lvl, features)
            if isinstance(r[0], list):
                res.extend(r)
            else:
                res.append(r)
        return res

    def _traverse_get_model(self, comp: GEMComposite):
        if comp.is_leaf:
            return comp
        res = []
        for c in comp:
            r = self._traverse_get_model(c)
            if isinstance(r, list):
                res.extend(r)
            else:
                res.append(r)
        return res

    def _traverse(self, tag=None, index=None, features=None):
        assert (tag is not None) ^ (index is not None)
        if tag is not None:
            comps = self.tget(tag)
        else:
            comps = self.iget(index)

        data = []
        for c in comps:
            if features is None:
                data += self._traverse_get_model(c)
            else:
                max_lvl = max([c._lvl for c in comps])
                data += self._traverse_util(c, [], max_lvl=max_lvl, features=features)
        return data

    def get_info(self, tag=None, index=None, features=None):
        if tag is None and index is None:
            tag = self.name_tag
        if features is None:
            features = ["n_rxns", "n_mets", "n_genes"]
        data = self._traverse(tag, index, features)
        col_names = [f"group_{i}" for i in range(len(data[0]) - len(features))] + features
        return pd.DataFrame(data=np.array(data), columns=col_names)

    def get_models(self, tag=None, index=None):
        if tag is None and index is None:
            tag = self.name_tag
        return self._traverse(tag, index, None)

    def summary(self):
        pass

    def plot_components(self,
                        group_order,
                        file_name: str = None):
        """
        Plot number of models' components

        Parameters
        ----------
        group_order: list of str
            Order of groups to plot the model component
        file_name: str
            Saved figure's file name, no plot will be saved if no filename is assigned.

        Examples
        ----------
        group = Group({"g1": ["model1", "model2"],
                       "g2": ["model3", "model4"],
                       "g3": ["model5", "model6"]})
        group.plot_components(group_order = ["g1", "g2", "g3"],
                              file_name = "model_components.png")
        group.plot_components(group_order = ["model1", "model2", "model3", "model4"],
                              file_name = "model_components.png")
        Returns
        -------

        """
        if group_order is None:
            group_order = list([g.name_tag for g in self._group])
        comp_df = self._component
        models_group = {mod.name: mod.group.name for mod in self._models}
        comp_df["samples"] = [s for s in comp_df.index]
        comp_df["groups"] = [models_group[s] for s in comp_df["samples"].to_list()]
        sample_grps = dict(zip(comp_df["samples"], comp_df["groups"]))
        new_comp_df = pd.concat(
            (pd.melt(comp_df, id_vars="samples", value_vars="reactions", var_name="component", value_name="number"),
             pd.melt(comp_df, id_vars="samples", value_vars="metabolites", var_name="component", value_name="number"),
             pd.melt(comp_df, id_vars="samples", value_vars="genes", var_name="component", value_name="number")),
            ignore_index=True
        )
        new_comp_df["group"] = new_comp_df["samples"].apply(lambda x: sample_grps[x])
        plot_model_components(new_comp_df, group_order, file_name=file_name)

    def plot_flux(self):
        pass

    def plot_flux_emb(self):
        pass

    def plot_flux_heatmap(self):
        pass

    def plot_expr_cluster(self):
        pass

    def plot_flux_cluster(self):
        pass

    def plot_model_heatmap(self, model_labels: Union[str, List[str]] = 'all',
                           components: Union[str, List[str]] = 'all',
                           all_model_level: bool = True,
                           annotate: bool = True,
                           fig_title: bool = None,
                           file_name=None,
                           prefix="heatmap_",
                           dpi=300,
                           **kwargs):
        """
        Plot the similarity of models' components

        Parameters
        ----------
        model_labels: str or a list of str

        components: str or a list of str
            if components is a list, use the names in the list to calculate similarity score.
            if 'all', use all of the components to calculate similarity score.
            Choose a category / categories from ['reactions', 'metabolites', 'genes']
        all_model_level
        annotate
        save_fig
        fig_title
        kwargs

        Returns
        -------

        """
        model_names = self.get_model_labels(model_labels)
        label_index = {label: ind for ind, label in enumerate(model_names)}
        jaccard_index = {f'{A}_to_{B}': self._cal_jaccard_index(A, B, components)
                         for A, B in itertools.combinations(model_names, 2)}
        jaccard_index.update({f'{A}_to_{A}': 1 for A in model_names})

        data = np.array([[jaccard_index[f'{sorted([A, B], key=lambda x: label_index[x])[0]}_to_{sorted([A, B], key=lambda x: label_index[x])[1]}']
                         for A in model_names]
                         for B in model_names])
        plot_heatmap(data=pd.DataFrame(data, index=model_names, columns=model_names),
                     xticklabels=True, yticklabels=True, scale=1,
                     cbar_label='Jaccard Index', cmap='magma', annotate=annotate,
                     fig_title=fig_title, file_name=file_name, prefix=prefix, dpi=dpi, **kwargs)



