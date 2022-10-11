from typing import List, Union, Dict
from functools import reduce
import itertools

import numpy as np
import pandas as pd
import cobra
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

import pipeGEM
from pipeGEM.core._base import GEMComposite
from pipeGEM.core._model import Model
from pipeGEM.plotting.categorical import plot_model_components
from pipeGEM.plotting.heatmap import plot_heatmap, plot_clustermap
from pipeGEM.plotting import plot_fba, plot_fva, plot_sampling
from pipeGEM.plotting.scatter import plot_PCA, plot_embedding
from pipeGEM.utils import is_iter, calc_jaccard_index
from pipeGEM.analysis import flux_analyzers


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
        self._group: np.ndarray = self._form_group(group)

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

    def __getitem__(self, item):
        for g in self._group:
            if g.name_tag == item:
                return g
        raise KeyError(item, " is not in the group")

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            self._group = np.append(self._group, [Group(group=value, name_tag=key)])
        elif isinstance(value, list):
            if all([isinstance(g, GEMComposite) for g in value]):
                self._group = np.append(self._group, value)
            else:
                ValueError("Input list must only contain Group or Model objects")
        elif isinstance(value, cobra.Model):
            self._group = np.append(self._group, [Model(model=value, name_tag=key)])
        elif isinstance(value, Model):
            self._group = np.append(self._group, [value])
        else:
            raise TypeError("Inputted value should be a dict, list, Model, or a cobra.model")

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

    def _get_group_model(self, group_strategy):
        if group_strategy == "top":
            return {g.name_tag: self._ravel_group(g) if not g.is_leaf else [g.name_tag] for g in self._group}

    def do_flux_analysis(self, method, aggregate_method="concat", solver="gurobi", group_strategy="top", **kwargs):
        results = []
        for c in self._group:
            if c.__class__ == self.__class__:
                result = c.do_flux_analysis(method=method, aggregate_method=aggregate_method,
                                            solver=solver, **kwargs)
            else:
                result = c.do_flux_analysis(method=method, solver=solver, **kwargs)
            results.append(result)
        return results[0].__class__.aggregate(results, method=aggregate_method,
                                              log={"name": self.name_tag,
                                                   "group": self._get_group_model(group_strategy)})

    def _ravel_group(self, comp_list):
        sel_models = []
        for i in comp_list:
            if i.is_leaf:
                sel_models.append(i)
            else:
                sel_models.extend(self._traverse_get_model(i))
        return sel_models

    def tget(self,
             tag: Union[str, list],
             ravel: bool = False,
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
        g = Group({'G1': {'model_1': model_1, 'model_2': model_2}})
        g.tget(['G1'])

        Returns
        -------
        selected_objects: list of Groups or Models
            A list of selected objects
        """
        if tag is None:
            selected = [self]
        elif isinstance(tag, str):
            selected = [g for g in self._group if g.name_tag == tag]
        elif isinstance(tag, list) or isinstance(tag, np.ndarray):
            selected = [g for g in self._group if g.name_tag in tag]
        else:
            raise ValueError
        if ravel:
            sel_models = self._ravel_group(selected)
            return self.__class__(sel_models, "selected_group")

        return self.__class__(selected, "selected_group")

    def iget(self,
             index,
             ravel: bool = False):
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

        Returns
        -------
        selected_objects: list
            A list of selected objects
        """
        if index is None:
            selected = [self]
        elif isinstance(index, int):
            selected = [self._group[index]]
        elif isinstance(index, int) or isinstance(index, list) or isinstance(index, np.ndarray):
            selected = self._group[index]
        else:
            raise ValueError
        if ravel:
            sel_models = self._ravel_group(selected)
            return self.__class__(sel_models, "selected_group")

        return self.__class__(selected, "selected_group")

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

    def _form_group(self, group_dict) -> np.ndarray:
        group_lis = []
        max_g = 0
        if isinstance(group_dict, list) or isinstance(group_dict, np.ndarray):
            if not all([isinstance(c, pipeGEM.Model) for c in group_dict]):
                raise ValueError("The input list should be a list of pipeGEM.Model")
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
        else:
            raise ValueError("Group doesn't support object type: ", type(group_dict))
        self._lvl = max_g + 1
        return np.array(group_lis)

    def _traverse_util(self, comp: GEMComposite, suffix_row, max_lvl, features, **f_kws) -> List:
        assert max_lvl >= len(suffix_row), f"{suffix_row}, {max_lvl}"
        if comp.is_leaf:
            return suffix_row + \
                   ["-" for _ in range(max_lvl - len(suffix_row))] + \
                   [getattr(comp, f)
                    if isinstance(getattr(type(comp), f), property)
                    else getattr(comp, f)(**f_kws)
                    for f in features]
        res = []
        assert is_iter(comp)
        for c in comp:
            r = self._traverse_util(c, suffix_row + [c.name_tag], max_lvl, features, **f_kws)
            if isinstance(r[0], list):
                res.extend(r)
            else:
                res.append(r)
        return res

    def _traverse_get_model(self, comp: GEMComposite) -> Union[List[GEMComposite], GEMComposite]:
        if comp.is_leaf:
            return comp
        res = []
        assert is_iter(comp)
        for c in comp:
            r = self._traverse_get_model(c)
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

        data = []
        for c in comps:
            max_lvl = max([c.tree_level for c in comps])
            data += self._traverse_util(c, [], max_lvl=max_lvl, features=features if features is not None else [],
                                        **kwargs)

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
        return pd.DataFrame(data=np.array(data), columns=col_names).infer_objects()

    def _find_by_nametag(self,
                         info_df: pd.DataFrame,
                         name_tag: str,
                         keep: str = "first",
                         ) -> Union[pd.Series, pd.DataFrame]:
        assert keep in ["first", "last", "all"]
        queries = [f"{c}=='{name_tag}'" for c in info_df.columns]
        res = info_df.query(" or ".join(queries))
        if keep == "first":
            res = res.iloc[:, 0]
        elif keep == "last":
            res = res.iloc[:, -1]
        return res

    def _get_by_tags(self, tags, get_model_level) -> List[GEMComposite]:
        if tags == "all":
            result = self._group
        elif isinstance(tags, str):
            result = self.tget(tags)
        elif isinstance(tags, list):
            result = []
            for t in tags:
                result.extend(self.tget(t))
        else:
            raise TypeError("")

        if get_model_level:
            models = []
            for r in result:
                if r.is_leaf:
                    models.append(r)
                else:
                    models.extend(self._traverse_get_model(r))
            return models
        return result

    def plot_components(self,
                        group_order = None,
                        file_name: str = None) -> pd.DataFrame:
        """
        Plot number of models' components and return the used df

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
        df: pd.DataFrame
            The dataframe used to generate the plot
        """
        if group_order is None:
            group_order = list([g.name_tag for g in self._group])
        comp_df = self.get_info(features=["n_rxns", "n_mets", "n_genes"])
        comp_df = comp_df.rename(columns={"group_0": "group", "group_1": "obj"})
        new_comp_df = pd.concat(
            (pd.melt(comp_df, id_vars=["group", "obj"], value_vars="n_rxns", var_name="component", value_name="number"),
             pd.melt(comp_df, id_vars=["group", "obj"], value_vars="n_mets", var_name="component", value_name="number"),
             pd.melt(comp_df, id_vars=["group", "obj"], value_vars="n_genes", var_name="component", value_name="number")),
            ignore_index=True
        )
        new_comp_df["number"] = new_comp_df["number"].astype(dtype=int)
        plot_model_components(new_comp_df, group_order, file_name=file_name)
        return new_comp_df

    def _process_flux(self,
                      method,
                      constr,
                      tags,
                      get_model_level,
                      aggregation_method,
                      show_groups = False
                      ) -> Dict[str, pd.DataFrame]:
        # TODO: remove

        if show_groups:
            compos: List[GEMComposite] = self._get_by_tags(tags, False)
        else:
            compos: List[GEMComposite] = self._get_by_tags(tags, get_model_level)
        # TODO: add more model info to fluxes result
        # compo_info = self.get_info(tags=tags if tags is not "all" else None)
        fluxes: Dict[str, Dict[str, pd.DataFrame]] = {}
        for c in compos:
            if c.is_leaf:
                flux = c.get_flux(method=method,
                                  constr=constr,
                                  as_dict=True,
                                  keep_rc=False)
            else:
                flux = c.get_flux(aggregate=aggregation_method if not show_groups else "concat",
                                  as_dict=True,
                                  method=method,
                                  constr=constr,
                                  keep_rc=False)
            for k, f in flux.items():
                if k not in fluxes:
                    fluxes[k] = {}
                processed = f.T
                processed["model"] = processed.index
                processed["group"] = c.name_tag
                fluxes[k][c.name_tag] = processed
        return {fname: pd.concat(list(fdfs.values()), axis=0).fillna(0) for fname, fdfs in fluxes.items() }

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

    @staticmethod
    def _compare_components(models,
                            components="all"):
        label_index = {model.name_tag: ind for ind, model in enumerate(models)}
        jaccard_index = {f'{A.name_tag}_to_{B.name_tag}': calc_jaccard_index(A, B, components)
                         for A, B in itertools.combinations(models, 2)}
        jaccard_index.update({f'{A.name_tag}_to_{A.name_tag}': 1 for A in models})
        model_names = [A.name_tag for A in models]
        result = np.array([[jaccard_index[f'{sorted([A, B], key=lambda x: label_index[x])[0]}'
                                          f'_to_'
                                          f'{sorted([A, B], key=lambda x: label_index[x])[1]}']
                          for A in model_names]
                          for B in model_names])
        result = pd.DataFrame(result, index=model_names, columns=model_names)

    def compare(self,
                tags,
                compare_models: bool = True,
                to_compare: str = "components",
                **kwargs
                ):
        models: List[GEMComposite] = self.tget(tags, compare_models)
        if to_compare == "components":
            self._compare_components(models=models, **kwargs)


    def plot_model_heatmap(self,
                           tags: Union[str, List[str]] = "all",
                           components: Union[str, List[str]] = 'all',
                           get_model_level: bool = True,
                           annotate: bool = True,
                           file_name=None,
                           prefix="model_jaccard_",
                           dpi=300,
                           **kwargs):
        # TODO: remove it
        """
        Plot the similarity of models' components

        Parameters
        ----------
        tags: str or list of str, default = 'all'
            Tag used to get the analyzed models, if input 'all', then get all the groups in this object
        components: str or a list of str
            if components is a list, use the names in the list to calculate similarity score.
            if 'all', use all of the components to calculate similarity score.
            Choose a category / categories from ['reactions', 'metabolites', 'genes']
        get_model_level: bool, default = True
            Whether to use model level only or use group level
        annotate: bool, default = True
            If to add annotation (number) on the heatmap
        file_name: str, optional, default = None
            output file name, if None then no file will be saved
        prefix: str, default = 'model_jaccard_'
            The file name prefix
        **kwargs

        Returns
        -------
        None
        """
        models: List[GEMComposite] = self._get_by_tags(tags, get_model_level)
        label_index = {model.name_tag: ind for ind, model in enumerate(models)}
        jaccard_index = {f'{A.name_tag}_to_{B.name_tag}': calc_jaccard_index(A, B, components)
                         for A, B in itertools.combinations(models, 2)}
        jaccard_index.update({f'{A.name_tag}_to_{A.name_tag}': 1 for A in models})
        model_names = [A.name_tag for A in models]
        data = np.array([[jaccard_index[f'{sorted([A, B],key=lambda x: label_index[x])[0]}'
                                        f'_to_'
                                        f'{sorted([A, B], key=lambda x: label_index[x])[1]}']
                         for A in model_names]
                         for B in model_names])

        plot_heatmap(data=pd.DataFrame(data, index=model_names, columns=model_names),
                     xticklabels=True,
                     yticklabels=True,
                     scale=1,
                     cbar_label='Jaccard Index',
                     cmap='magma',
                     annotate=annotate,
                     file_name=file_name,
                     prefix=prefix,
                     dpi=dpi,
                     **kwargs)

