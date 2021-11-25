from typing import List, Union, Dict
from functools import reduce
import itertools

import numpy as np
import pandas as pd
import cobra
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

from pipeGEM.core._base import GEMComposite
from pipeGEM.core._model import Model
from pipeGEM.plotting.categorical import plot_model_components
from pipeGEM.plotting.heatmap import plot_heatmap, plot_clustermap
from pipeGEM.plotting import plot_fba, plot_fva, plot_sampling
from pipeGEM.plotting.scatter import plot_PCA, plot_embedding
from pipeGEM.utils import is_iter, calc_jaccard_index


class Group(GEMComposite):
    _is_leaf = False
    agg_methods = ["concat", "mean", "sum", "min", "max", "weighted_mean", "absmin", "absmax"]

    def __init__(self,
                 group,
                 name_tag: str = None,
                 data=None):
        """
        Main container for performing model comparison

        Parameters
        ----------
        group
        name_tag
        data
        """
        super().__init__(name_tag=name_tag)
        self.data = data
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

    def __getitem__(self, item):
        for g in self._group:
            if g.name_tag == item:
                return g

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            self._group.append(Group(group=value, name_tag=key))
        elif isinstance(value, list):
            if all([isinstance(g, GEMComposite) for g in value]):
                self._group.extend(value)
            else:
                ValueError("Input list must only contain Group or Model objects")
        elif isinstance(value, cobra.Model):
            self._group.append(Model(model=value, name_tag=key))
        elif isinstance(value, Model):
            self._group.append(value)
        else:
            raise TypeError("Inputted value should be a dict, list, Model, or a cobra.model")

    @property
    def reaction_ids(self):
        return list(reduce(set.union, [set(g.reaction_ids) for g in self._group]))

    @property
    def metabolite_ids(self):
        return list(reduce(set.union, [set(g.metabolite_ids) for g in self._group]))

    @property
    def gene_ids(self):
        return list(reduce(set.union, [set(g.gene_ids) for g in self._group]))

    @property
    def size(self):
        return sum([g.size for g in self._group])

    @property
    def members(self):
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

    def do_analysis(self, **kwargs):
        for g in self._group:
            g.do_analysis(**kwargs)

    def get_flux(self, aggregate="concat", as_dict=True, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Get flux dataframe in this object,
        The output of this function could be a dataframe(FBA, pFBA, sampling), or two dataframes (FVA)
        The shape of the dataframes also depend on the method and aggregation method users specified,
        method -
            FBA or pFBA: return a dict contains a dataframe with shape = (n_rxns, n_models) when aggregate = 'concat',
                otherwise the shape will be (n_rxns, 1)
            FVA: return a dict contains two dataframes with shape = (n_rxns, n_models) when aggregate = 'concat',
                otherwise the shape of each frame will be (n_rxns, 1)
            sampling: return a dict contains n dataframes with shape = (n_rxns, n_models) when aggregate = 'concat',
                otherwise the shape of each frame will be (n_rxns, 1)

        Parameters
        ----------
        as_dict
        aggregate: str
            The name of aggregation method, choose from: ["concat", "mean", "sum", "min", "max", "weighted_mean"]

        kwargs

        Returns
        -------

        """
        if aggregate not in self.agg_methods:
            raise ValueError("This aggregation method is not exist")

        dfs: Dict[str, Dict[str,
                            Union[pd.DataFrame, pd.Series]]] = \
            {g.name_tag: g.get_flux(as_dict=as_dict, **kwargs) for g in self._group}
        col_names = list(list(dfs.values())[0].keys())
        grouped_dfs = {c: pd.concat([dfs[g.name_tag][c] for g in self._group], axis=1)
                       for c in col_names}

        # TODO: fix - violate open-close principle
        if aggregate in ["mean", "sum", "min", "max", "weighted_mean", "absmin", "absmax"]:
            for k, v in grouped_dfs.items():
                first_agg = {a: a for a in self.agg_methods}
                first_agg.update({"weighted_mean": "sum", "absmin": "abs", "absmax": "abs"})
                if aggregate in ["absmin", "absmax"]:
                    grouped_dfs[k] = getattr(v, first_agg[aggregate])()
                else:
                    grouped_dfs[k] = getattr(v, first_agg[aggregate])(axis=1)
                grouped_dfs[k] = grouped_dfs[k].to_frame()
                grouped_dfs[k].columns = [self.name_tag]
                if aggregate == "weighted_mean":
                    grouped_dfs[k] = grouped_dfs[k] / [g.size for g in self._group]
                elif aggregate == "absmin":
                    grouped_dfs[k] = grouped_dfs[k].min()
                elif aggregate == "absmax":
                    grouped_dfs[k] = grouped_dfs[k].max()
        return grouped_dfs

    def tget(self,
             tag: Union[str, list]) -> List[GEMComposite]:
        """
        Use name_tag to find groups or models in this group

        Parameters
        ----------
        tag: str, list or None
            tag could be different types:
                None - get all of the group objects in this group
                str - get the group objs whose name_tag match the tag in this group
                list - sequentially find the matched objects

        Examples
        ----------
        g = Group({'G1': {'model_1': model_1, 'model_2': model_2}})
        g.tget(['G1', 'model_1'])

        Returns
        -------
        selected_objects: list of Groups or Models
            A list of selected objects
        """
        if tag is None:
            selected = [self]
        elif isinstance(tag, str):
            selected = [g for g in self._group if g.name_tag == tag]
        elif isinstance(tag, list):
            if len(tag) > 1:
                selected = [g.tget(tag[1:]) if not g.is_leaf else g for g in self._group if g.name_tag == tag[0]]
            else:
                selected = self.tget(tag[0])
        else:
            raise ValueError
        return selected

    def iget(self, index) -> List[GEMComposite]:
        """
        Use index to find groups or models in this group

        Parameters
        ----------
        index: int, list or None
            index could be different types:
                None - get all of the group objects in this group
                int - get the group objs whose index match the index in this group
                list - sequentially find the matched objects
        Returns
        -------
        selected_objects: list
            A list of selected objects
        """
        if index is None:
            selected = [self]
        elif isinstance(index, int):
            selected = self._group[index]
        elif isinstance(index, list):
            if len(index) > 1:
                selected = [g.iget(index[1:]) if not g.is_leaf else g for g in self._group[index[0]]]
            else:
                selected = self.iget(index[0])
        else:
            raise ValueError
        return selected

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
        for name, comp in group_dict.items():
            if isinstance(comp, dict):
                g = Group(group=comp, name_tag=name, data=self.data)
                max_g = max(max_g, g.tree_level)
                group_lis.append(g)
            else:
                group_lis.append(Model(model=comp, name_tag=name, data=self.data))
                max_g = max(max_g, 0)
        self._lvl = max_g + 1
        return group_lis

    def _traverse_util(self, comp: GEMComposite, suffix_row, max_lvl, features, **f_kws) -> List:
        assert max_lvl >= len(suffix_row)
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

    def get_info(self, tag=None, index=None, features=None, **kwargs) -> pd.DataFrame:
        """

        Parameters
        ----------
        tag
        index
        features
        kwargs

        Returns
        -------

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
                        group_order,
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
        found_objs: Dict[str, pd.Series] = {g: self._find_by_nametag(comp_df, g) for g in group_order}
        obj_parents = {name: g.iloc[g[g == name].index[0]-1] for name, g in found_objs.items()}
        comp_df["obj"] = group_order
        comp_df["groups"] = [obj_parents[n] for n in comp_df["obj"].to_list()]
        sample_grps = dict(zip(comp_df["obj"], comp_df["groups"]))
        new_comp_df = pd.concat(
            (pd.melt(comp_df, id_vars="obj", value_vars="n_rxns", var_name="component", value_name="number"),
             pd.melt(comp_df, id_vars="obj", value_vars="n_mets", var_name="component", value_name="number"),
             pd.melt(comp_df, id_vars="obj", value_vars="n_genes", var_name="component", value_name="number")),
            ignore_index=True
        )
        new_comp_df["number"] = new_comp_df["number"].astype(dtype=int)
        new_comp_df["group"] = new_comp_df["obj"].apply(lambda x: sample_grps[x])
        plot_model_components(new_comp_df, group_order, file_name=file_name)
        return new_comp_df

    def _process_flux(self,
                      method,
                      constr,
                      tags,
                      get_model_level,
                      aggregation_method,
                      ) -> Dict[str, pd.DataFrame]:
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
                flux = c.get_flux(aggregate=aggregation_method,
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
        return {fname: pd.concat(list(fdfs.values()), axis=0) for fname, fdfs in fluxes.items() }

    def plot_flux(self,
                  method,
                  constr,
                  rxn_ids = None,
                  rxn_index = None,
                  subsystems = None,
                  tags: Union[str, List[str]] = "all",
                  get_model_level: bool = True,
                  aggregation_method="mean",
                  **kwargs
                  ):
        """
        Plot the flux analysis results stored in the model

        Parameters
        ----------
        method
        constr
        rxn_ids
        rxn_index
        subsystems
        tags
        get_model_level
        aggregation_method
        kwargs

        Returns
        -------

        """
        rxn_ids = rxn_ids if rxn_ids is not None else []
        rxn_ids += self._check_rxn_id(self.tget(tags if tags != "all" else None)[0], rxn_index, subsystems)
        fluxes = self._process_flux(method, constr, tags, get_model_level, aggregation_method)
        if method in ["FBA", "pFBA"]:
            plot_fba(flux_df=fluxes["fluxes"], rxn_ids=rxn_ids, group_layer="group", **kwargs)
        elif method == "FVA":
            if aggregation_method == ["concat", "sum"]:
                raise ValueError("This aggregation method is not appropriate, choose from mean, absmin, absmax")
            plot_fva(min_flux_df=fluxes["minimum"],
                     max_flux_df=fluxes["maximum"], rxn_ids=rxn_ids, group_layer="group", **kwargs)
        elif method == "sampling":
            plot_sampling(sampling_flux_df=fluxes, rxn_ids=rxn_ids, group_layer="group", **kwargs)
        else:
            raise NotImplementedError()

    def plot_flux_emb(self,
                      method,
                      constr,
                      dr_method="PCA",
                      rxn_ids="all",
                      rxn_index=None,
                      subsystems=None,
                      tags: Union[str, List[str]] = "all",
                      aggregation_method="mean",
                      **kwargs
                      ):
        get_model_level = True
        # rxn_ids = rxn_ids if rxn_ids is not None else []
        # rxn_ids += self._check_rxn_id(self.tget(tags if tags != "all" else None)[0], rxn_index, subsystems)
        fluxes = self._process_flux(method, constr, tags, get_model_level, aggregation_method)
        if method in ["FBA", "pFBA"]:
            df: pd.DataFrame = fluxes["fluxes"]
        elif method == "sampling":
            df: pd.DataFrame = pd.concat(list(fluxes.values()), axis=0)
        else:
            raise NotImplementedError()
        groups = dict(df["group"].to_frame().reset_index(drop=True).reset_index().groupby("group")["index"].apply(list).iteritems())
        df = df.drop(columns=["model", "group"]).reset_index(drop=True)
        if dr_method == "PCA":
            plot_PCA(df=df.T, groups=groups, **kwargs)
        else:
            plot_embedding(df=df.T, reducer=dr_method, groups=groups, **kwargs)

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
                          **kwargs
                          ):

        similarity_method = {'cosine': cosine_similarity,
                             'euclidean': lambda x: 1 - euclidean_distances(x) / np.amax(euclidean_distances(x)),
                             'manhattan': lambda x: 1 - manhattan_distances(x) / np.amax(manhattan_distances(x))}
        # rxn_ids = rxn_ids if rxn_ids is not None else []
        # rxn_ids += self._check_rxn_id(self.tget(tags if tags != "all" else None)[0], rxn_index, subsystems)
        fluxes = self._process_flux(method, constr, tags, get_model_level, aggregation_method)
        print(fluxes["fluxes"])
        if method in ["FBA", "pFBA"]:
            model_names = fluxes["fluxes"]["model"] if not get_model_level else fluxes["fluxes"]["group"]
            comp_info = dict(zip(fluxes["fluxes"]["model"], fluxes["fluxes"]["group"])) \
                        if not get_model_level else fluxes["fluxes"]["group"]
            data = fluxes["fluxes"].drop(columns=["model", "group"] if get_model_level else ["group"]).values
        else:
            raise ValueError()

        data = pd.DataFrame(data=similarity_method[similarity](data),
                            columns=model_names,
                            index=model_names)
        print(data)
        # xticklabels = model_names,
        # yticklabels = model_names,
        plot_clustermap(data=data,
                        cbar_label=f'{similarity} similarity',
                        cmap='magma',
                        square=True,
                        fig_size=fig_size,
                        **kwargs
                        )

    def plot_expr_cluster(self):
        pass

    def plot_expr_heatmap(self):
        pass

    def plot_model_heatmap(self,
                           tags: Union[str, List[str]] = "all",
                           components: Union[str, List[str]] = 'all',
                           get_model_level: bool = True,
                           annotate: bool = True,
                           file_name=None,
                           prefix="model_jaccard_",
                           dpi=300,
                           **kwargs):
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
                     dpi=dpi, **kwargs)



