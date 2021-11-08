from typing import List, Union, Dict
from functools import reduce
import itertools

import numpy as np
import pandas as pd
import cobra

from pipeGEM.core._base import GEMComposite
from pipeGEM.core._model import Model
from pipeGEM.plotting.categorical import plot_model_components
from pipeGEM.plotting.heatmap import plot_heatmap
from pipeGEM.plotting._flux import plot_fba, plot_fva
from pipeGEM.utils import is_iter, calc_jaccard_index


class Group(GEMComposite):
    _is_leaf = False

    def __init__(self,
                 group,
                 name_tag: str = None,
                 data=None):
        """

        Parameters
        ----------
        group
        name_tag
        data
        """
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

    def get_flux(self, aggregate="concat", **kwargs) -> pd.DataFrame:
        if aggregate not in ["concat", "mean", "sum", "min", "max", "weighted_mean"]:
            raise ValueError("This aggregation method is not exist")

        dfs: Dict[str, pd.DataFrame] = {g.name_tag: g.get_flux(**kwargs) for g in self._group}
        col_name_dic = {c: f"{k}_{c}" for k, v in dfs.items() for c in v.columns }
        dfs = {k: v.rename(columns=col_name_dic) for k, v in dfs.items()}
        result = pd.concat(list(dfs.values()), axis=1)
        if aggregate in ["mean", "sum", "min", "max"]:
            result = result.T
            label_dic = {v: k for k, v in dfs.items()}
            result["label"] = result.index.map(label_dic)
            gp = result.groupby("label")
            result = getattr(gp, aggregate).T
        return result

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

    def _traverse_util(self, comp: GEMComposite, suffix_row, max_lvl, features, f_kws: dict) -> List:
        assert max_lvl >= len(suffix_row)
        if comp.is_leaf:
            return suffix_row + \
                   ["-" for _ in range(max_lvl - len(suffix_row))] + \
                   [getattr(comp, f)
                    if isinstance(getattr(type(comp), f), property)
                    else getattr(comp, f)(**f_kws.get(f, default={}))
                    for f in features]
        res = []
        assert is_iter(comp)
        for c in comp:
            r = self._traverse_util(c, suffix_row + [c.name_tag], max_lvl, features, f_kws)
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

    def _traverse(self, tag=None, index=None, features=None):
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

    def get_info(self, tag=None, index=None, features=None) -> pd.DataFrame:
        data = self._traverse(tag, index, features)
        features = ["model_obj"] if features is None else features
        col_names = [f"group_{i}" for i in range(len(data[0]) - len(features))] + features
        return pd.DataFrame(data=np.array(data), columns=col_names)

    def _find_by_nametag(self,
                         info_df: pd.DataFrame,
                         name_tag: str,
                         keep: str = "first",
                         ) -> Union[pd.Series, pd.DataFrame]:
        assert keep in ["first", "last", "all"]
        queries = [f"{c}=='{name_tag}" for c in info_df.columns]
        res = info_df.query(" or ".join(queries))
        if keep == "first":
            res = res.iloc[:, 0]
        elif keep == "last":
            res = res.iloc[:, -1]
        return res

    def _get_by_tags(self, tags, get_model_level) -> List[GEMComposite]:
        if tags == "all":
            result = self._group
        if isinstance(tags, str):
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
        new_comp_df["group"] = new_comp_df["obj"].apply(lambda x: sample_grps[x])
        plot_model_components(new_comp_df, group_order, file_name=file_name)
        return new_comp_df

    def plot_flux(self,
                  method,
                  constr,
                  tags: Union[str, List[str]] = "all",
                  get_model_level: bool = True,
                  aggregation_method="mean",
                  **kwargs
                  ):
        compos: List[GEMComposite] = self._get_by_tags(tags, get_model_level)
        fluxes = {c.name_tag: c.get_flux(aggregate=aggregation_method,
                                         method=method, constr=constr, keep_rc=False)
                  for c in compos}
        if method in ["FBA", "pFBA"]:
            if aggregation_method == "concat":
                plot_fba(flux_df=pd.concat(list(fluxes.values()), axis=1), **kwargs)
            else:
                fs = []
                for name, f_df in fluxes.items():
                    fs.append(f_df.rename(columns={"fluxes": name}))
                plot_fba(flux_df=pd.concat(fs, axis=1), **kwargs)
        else:
            raise NotImplementedError()  # TODO: finish

    def plot_flux_emb(self):
        pass

    def plot_flux_cluster(self):
        pass

    def plot_flux_heatmap(self):
        pass

    def plot_expr_cluster(self):
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



