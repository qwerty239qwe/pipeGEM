from typing import List, Union, Optional
from functools import reduce
import itertools

import numpy as np
import pandas as pd
import cobra
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

import pipeGEM
from pipeGEM.core._base import GEMComposite
from pipeGEM.core._model import Model
from pipeGEM.analysis import ComponentComparisonAnalysis, ComponentNumberAnalysis
from pipeGEM.data import GeneData
from pipeGEM.plotting import plot_clustermap
from pipeGEM.utils import is_iter, calc_jaccard_index


class Group(GEMComposite):
    _is_leaf = False
    agg_methods = ["concat", "mean", "sum", "min", "max", "weighted_mean", "absmin", "absmax"]

    def __init__(self,
                 group,
                 name_tag: str = None):
        """
        Main container for performing model comparison

        Parameters
        ----------
        group: a list of pg.Model or a dict of dicts
            The name_tag and models used to build the pg.Group object,
            Possible inputs are: {name_tag of subgroup: [pg.Models]} and
            {name_tag of subgroup: {name_tag of model: cobra.Model}}
        name_tag: optional, str
            The name of this object
        """
        super().__init__(name_tag=name_tag)
        self._group: list = self._form_group(group)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Group [{self._name_tag}]" + self._next_layer_tree()

    def _next_layer_tree(self):
        comps = [f"[{i}] {comp.name_tag} ({'Model)' if comp.is_leaf else 'Group) ── ' + str(comp.size)}"
                 for i, comp in enumerate(self._group)]
        result = "\n├── "
        if len(comps) > 1:
            result += "\n├── ".join(comps[:-1])
        if len(comps) != 0:
            result += "\n└── " + comps[-1]
        return result

    def __len__(self):
        return len(self._group)

    def __iter__(self):
        index = 0
        while index < len(self._group):
            yield self._group[index]
            index += 1

    def __contains__(self, item):
        for g in self._group:
            if g.name_tag == item:
                return True
        return item in self._group

    def __getitem__(self, item):
        for g in self._group:
            if g.name_tag == item:
                return g
        raise KeyError(item, " is not in the group")

    def __setitem__(self, key, value):

        if isinstance(value, dict):
            self._group.append(self.__class__(group=value, name_tag=key))
        elif isinstance(value, list):
            if all([isinstance(g, GEMComposite) for g in value]):
                self._group.append(self.__class__(group=value, name_tag=key))
            else:
                ValueError("Input list must only contain Group or Model objects")
        elif isinstance(value, cobra.Model):
            self._group.append(Model(model=value, name_tag=key))
        elif isinstance(value, GEMComposite):
            value._name_tag = key
            self._group.append(value)
        else:
            raise TypeError("Inputted value should be a dict, list, Model, or a cobra.model")

    def index(self, item, raise_err=True):
        for i, g in enumerate(self._group):
            if g.name_tag == item:
                return i
        if raise_err:
            raise KeyError(item, " is not in this group")

    @property
    def reaction_ids(self) -> List[str]:
        """
        All the reaction ID inside this object

        Returns
        -------
        rxn_ids: list of str
        """
        return list(reduce(set.union, [set(g.reaction_ids) for g in self._group]))

    @property
    def metabolite_ids(self):
        """
        All the metabolites ID inside this object

        Returns
        -------
        met_ids: list of str
        """
        return list(reduce(set.union, [set(g.metabolite_ids) for g in self._group]))

    @property
    def gene_ids(self):
        """
        All the genes ID inside this object

        Returns
        -------
        gene_ids: list of str
        """
        return list(reduce(set.union, [set(g.gene_ids) for g in self._group]))

    @property
    def size(self) -> int:
        """
        The size of models in this object

        Returns
        -------
        size: int
        """
        return sum([g.size for g in self._group])

    @property
    def members(self) -> str:
        """
        Show all of the members in this objects

        Returns
        -------
        members: str
        """
        return "\n".join([str(g) for g in self._group])

    @property
    def subsystems(self):
        subs = {}
        for g in self._group:
            for s_name, rxns in g.subsystems.items():
                if s_name in subs:
                    subs[s_name] += rxns
                else:
                    subs[s_name] = rxns
        for g, rxns in subs.items():
            subs[g] = set(rxns)
        return subs

    @property
    def gene_data(self):
        models = self.tget(None, ravel=True)
        gene_data = {m.name_tag: m.gene_data for m in models}
        return GeneData.aggregate(gene_data, prop="data")

    def get_RAS(self, data_name, method="mean"):
        models = self.tget(None, ravel=True)
        gene_data = {m.name_tag: m.gene_data for m in models}
        return GeneData.aggregate(gene_data, prop="data")

    def _get_group_model(self, group_strategy):
        if group_strategy == "top":
            return {g.name_tag: self._ravel_group(g, "name_tag") if not g.is_leaf else [g.name_tag] for g in self._group}

    def do_flux_analysis(self, method, aggregate_method="concat", solver="gurobi", group_strategy="top", **kwargs):
        results = []
        for c in self._group:
            if c.__class__ == self.__class__:
                result = c.do_flux_analysis(method=method, aggregate_method=aggregate_method,
                                            solver=solver, **kwargs)
            else:
                result = c.do_flux_analysis(method=method, solver=solver, **kwargs)
                result.add_name(c.name_tag)
            results.append(result)
        return results[0].__class__.aggregate(results, method=aggregate_method,
                                              log={"name": self.name_tag,
                                                   "group": self._get_group_model(group_strategy)})

    def _ravel_group(self, comp_list, get_obj="object"):
        sel_models = []
        for i in comp_list:
            if i.is_leaf:
                if get_obj == "object":
                    sel_models.append(i)
                else:
                    sel_models.append(getattr(i, get_obj))
            else:
                sel_models.extend(self._traverse_get_model(i, get_obj=get_obj))
        return sel_models

    def tget(self,
             tag: Optional[Union[str, list]] = None,
             ravel: bool = False,
             name: str = "selected_group"
             ):
        """
        Use name_tag to find groups or models in this group

        Parameters
        ----------
        tag: str, list or None
            tag could be different types:
                None - get all of the group objects in this group
                str - get the group objs whose name_tag match the tag in this group
                list - get all the matched objects

        ravel: bool
            Return a group with only models

        Examples
        ----------
        g = Group({'G1': {'model_1': model_1, 'model_2': model_2}, 'G2': {'model_3': model_3}})
        g1 = g.tget(['G1'])

        Returns
        -------
        selected_group: Group
            A Group containing the selected objects
        """
        if tag is None:
            selected = [self]
            name = self.name_tag
        elif isinstance(tag, str):
            selected = [g for g in self._group if g.name_tag == tag]
        elif isinstance(tag, list) or isinstance(tag, np.ndarray):
            selected = [g for g in self._group if g.name_tag in tag]
        else:
            raise ValueError
        if ravel:
            sel_models = self._ravel_group(selected)
            return self.__class__(sel_models, name)

        if len(selected) == 1:
            return selected[0]
        return self.__class__(selected, name)

    def iget(self,
             index,
             ravel: bool = False,
             name: str = "selected_group"):
        """
        Use index to find groups or models in this group

        Parameters
        ----------
        index: int, list or None
            index could be different types:
                None - get all of the group objects in this group
                int - get the group objs whose index match the index in this group
                list - sequentially find the matched objects

        ravel: bool
            Return a group with only models
        name: str
            Name of returned Group

        Returns
        -------
        selected_group: Group
            A Group containing the selected objects
        """
        if index is None:
            selected = [self]
            name = self.name_tag
        elif isinstance(index, int):
            selected = [self._group[index]]
            name = self._group[index].name_tag
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            selected = [self._group[i] for i in index]
        else:
            raise ValueError
        if ravel:
            sel_models = self._ravel_group(selected)
            return self.__class__(sel_models, name)

        if len(selected) == 1:
            return selected[0]
        return self.__class__(selected, name)

    @staticmethod
    def _check_rxn_id(comp: GEMComposite, index, subsystems):
        if index is None and subsystems is None:
            return []
        if index is not None:
            return [comp.reaction_ids[r] for r in index]
        if subsystems is not None:
            all = []
            for s in subsystems:
                all += comp.subsystems[s]
            return all

    def _form_group(self, group_dict) -> list:
        group_lis = []
        max_g = 0
        if isinstance(group_dict, list) or isinstance(group_dict, np.ndarray):
            if not all([isinstance(c, GEMComposite) for c in group_dict]):
                raise ValueError("The input list should be a list of pipeGEM.Model, the list contains types: ",
                                 set([type(c) for c in group_dict]))
            group_lis = [c for c in group_dict]
            max_g = max([g.tree_level for g in group_lis] + [1])
        elif isinstance(group_dict, dict):
            for name, comp in group_dict.items():
                if isinstance(comp, dict):
                    g = Group(group=comp, name_tag=name)
                    max_g = max(max_g, g.tree_level)
                    group_lis.append(g)
                elif isinstance(comp, list) or isinstance(comp, np.ndarray):
                    group_lis.extend(list(comp))
                    max_g = max([c.tree_level for c in group_lis])
                elif isinstance(comp, cobra.Model):
                    group_lis.append(Model(model=comp, name_tag=name))
                    max_g = max([g.tree_level for g in group_lis] + [1])
                elif isinstance(comp, pipeGEM.Model):
                    group_lis.append(comp)
                    max_g = max([g.tree_level for g in group_lis] + [1])
        elif group_dict is None:
            group_lis = []
        else:
            raise ValueError("Group doesn't support object type: ", type(group_dict))
        self._lvl = max_g + 1
        return group_lis

    def _traverse_util(self, comp: GEMComposite, data, suffix_row, row_idx, features, **f_kws) -> int:
        if comp.is_leaf:
            inserted = suffix_row + [comp.name_tag] + \
                       ["-" for _ in range(data.shape[1] - len(suffix_row) - 1 - len(features))] + \
                       [getattr(comp, f)
                        if isinstance(getattr(type(comp), f), property) else getattr(comp, f)(**f_kws)
                        for f in features]
            data[row_idx, :] = inserted

            return row_idx + 1
        assert is_iter(comp)
        for c in comp:
            row_idx = self._traverse_util(c, data, suffix_row + [comp.name_tag], row_idx, features, **f_kws)
        return row_idx

    def _traverse_get_model(self, comp: GEMComposite, get_obj="object") -> Union[List[GEMComposite], GEMComposite]:
        if comp.is_leaf:
            if get_obj == "object":
                return comp
            else:
                return getattr(comp, get_obj)
        res = []
        assert is_iter(comp)
        for c in comp:
            r = self._traverse_get_model(c, get_obj=get_obj)
            if isinstance(r, list):
                res.extend(r)
            else:
                res.append(r)
        return res

    def _traverse(self, tag=None, index=None, features=None, **kwargs):
        if tag is not None:
            comps = self.tget(tag)
        else:
            comps = self.iget(index)

        data = np.empty(shape=(comps.size, self.tree_level + (len(features) if features is not None else 0)), dtype="O")
        row_idx = self._traverse_util(comps, data=data, suffix_row=[],
                                      row_idx=0, features=features if features is not None else [],
                                      **kwargs)
        assert row_idx == data.shape[0]
        return data

    def get_info(self,
                 tag=None,
                 index=None,
                 features=None,
                 **kwargs) -> pd.DataFrame:
        """
        Get a information table by traversing the object structure

        Parameters
        ----------
        tag: optional, str
            The name tag of the root object, if None, use this object as the root.
            Cannot input with index simultaneously
        index: optional, int
            The index of the root object, if None, use this object as the root.
            Cannot input with tag simultaneously
        features: optional, list of str
            The features to be obtained while the traverse
        kwargs: dict
            Keyword args used while traversing

        Returns
        -------
        information_table: pd.DataFrame

        """
        data = self._traverse(tag, index, features, **kwargs)
        features = features if features is not None else []
        col_names = [f"group_{i}" for i in range(len(data[0]) -
                                                 len(features))] + features
        return pd.DataFrame(data=data, columns=col_names).infer_objects()

    @staticmethod
    def _compare_components(models,
                            components="all"):
        label_index = {model.name_tag: ind for ind, model in enumerate(models)}
        jaccard_index = {f'{A.name_tag}_to_{B.name_tag}': calc_jaccard_index(A, B, components)
                         for A, B in itertools.combinations(models, 2)}
        jaccard_index.update({f'{A.name_tag}_to_{A.name_tag}': 1 for A in models})
        model_names = [A.name_tag for A in models]
        comp_arr = np.array([[jaccard_index[f'{sorted([A, B], key=lambda x: label_index[x])[0]}'
                                          f'_to_'
                                          f'{sorted([A, B], key=lambda x: label_index[x])[1]}']
                          for A in model_names]
                          for B in model_names])
        comp_arr = pd.DataFrame(comp_arr, index=model_names, columns=model_names)
        result = ComponentComparisonAnalysis(log={"components": components})
        result.add_result(comp_arr)
        return result

    @staticmethod
    def _compare_component_num(models,
                               components="all",
                               name_order: Union[str, list]="default",
                               present_lvl: int = 1):
        components = ['genes', 'reactions', 'metabolites'] if components == "all" else components
        c_to_f = {"genes": "n_genes", "reactions": "n_rxns", "metabolites": "n_mets"}
        comp_df = models.get_info(features=[c_to_f[c] for c in components])
        comp_df = comp_df.rename(columns={f"group_{present_lvl-1}": "group", f"group_{present_lvl}": "model"})
        new_comp_df = pd.concat(
            (pd.melt(comp_df, id_vars=["group", "model"], value_vars="n_rxns", var_name="component", value_name="number"),
             pd.melt(comp_df, id_vars=["group", "model"], value_vars="n_mets", var_name="component", value_name="number"),
             pd.melt(comp_df, id_vars=["group", "model"], value_vars="n_genes", var_name="component",
                     value_name="number")),
            ignore_index=True
        )

        new_comp_df["number"] = new_comp_df["number"].astype(dtype=int)
        result = ComponentNumberAnalysis(log={"components": components, 'present_lvl': present_lvl})
        result.add_result(new_comp_df, name_order=name_order
                                                  if name_order != "default" else None)
        return result

    def compare(self,
                tags=None,
                compare_models: bool = True,
                use: str = "jaccard",
                **kwargs
                ):
        assert use in ["jaccard", "PCA", "num"]

        models: Group = self.tget(tags, compare_models)
        if use == "jaccard":
            return self._compare_components(models=models, **kwargs)
        elif use == "num":
            return self._compare_component_num(models=models, name_order=[models.name_tag], **kwargs)




    def plot_flux_heatmap(self,
                          method,
                          constr,
                          similarity="cosine",
                          rxn_ids="all",
                          rxn_index=None,
                          subsystems=None,
                          tags: Union[str, List[str]] = "all",
                          get_model_level=True,
                          aggregation_method="mean",
                          fig_size=(10, 10),
                          file_name=None,
                          **kwargs
                          ):

        similarity_method = {'cosine': cosine_similarity,
                             'euclidean': lambda x: 1 - euclidean_distances(x) / np.amax(euclidean_distances(x)),
                             'manhattan': lambda x: 1 - manhattan_distances(x) / np.amax(manhattan_distances(x))}
        # rxn_ids = rxn_ids if rxn_ids is not None else []
        # rxn_ids += self._check_rxn_id(self.tget(tags if tags != "all" else None)[0], rxn_index, subsystems)
        fluxes = self._process_flux(method, constr, tags, get_model_level, aggregation_method)
        if method in ["FBA", "pFBA"]:
            model_names = fluxes["fluxes"]["model"] if not get_model_level else fluxes["fluxes"]["group"]
            comp_info = dict(zip(fluxes["fluxes"]["model"], fluxes["fluxes"]["group"])) \
                        if not get_model_level else fluxes["fluxes"]["group"]
            data = fluxes["fluxes"].drop(columns=["model", "group"] if get_model_level else ["group"]).fillna(0).values
        else:
            raise ValueError()
        data = pd.DataFrame(data=similarity_method[similarity](data),
                            columns=model_names,
                            index=model_names)
        plot_clustermap(data=data,
                        cbar_label=f'{similarity} similarity',
                        cmap='magma',
                        square=True,
                        fig_size=fig_size,
                        file_name=file_name,
                        **kwargs
                        )

    def plot_expr_heatmap(self,
                          tags: Union[str, List[str]] = "all",
                          get_model_level=True,
                          aggregation_method="mean",
                          fig_size=(10, 10),
                          file_name=None,
                          **kwargs
                          ):
        # TODO: replace it

        data = self.data
        plot_clustermap(data=data,
                        cbar_label=f'expression',
                        cmap='magma',
                        square=True,
                        fig_size=fig_size,
                        file_name=file_name,
                        **kwargs
                        )
