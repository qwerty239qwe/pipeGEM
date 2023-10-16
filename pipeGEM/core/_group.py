from typing import List, Union, Optional, Tuple
from functools import reduce
import itertools
from warnings import warn

import numpy as np
import pandas as pd
import cobra
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

from pipeGEM.core._base import GEMComposite
from pipeGEM.core._model import Model
from pipeGEM.analysis import ComponentComparisonAnalysis, ComponentNumberAnalysis, prepare_PCA_dfs, PCA_Analysis
from pipeGEM.data import GeneData
from pipeGEM.plotting import plot_clustermap
from pipeGEM.utils import is_iter, calc_jaccard_index


class Group(GEMComposite):
    """
    Main container for performing model comparison

    Parameters
    ----------
    group: a list of pg.Model or a dict of dicts
        The name_tag and models used to build the pg.Group object,
        Possible inputs are:
        [pg.Models],
        {name_tag of model: cobra.Model},
        {name_tag of subgroup: [pg.Models]}, and
        {name_tag of subgroup: {name_tag of model: cobra.Model}}
    name_tag: optional, str
        The name of this object. If None, this group will be named 'Unnamed_group'
    """
    _is_leaf = False
    agg_methods = ["concat", "mean", "sum", "min", "max", "weighted_mean", "absmin", "absmax"]

    def __init__(self,
                 group,
                 name_tag: str = None,
                 factors: pd.DataFrame = None,
                 **kwargs):

        super().__init__(name_tag=name_tag or "Unnamed_group")
        self._group_annotation = {}
        self._group = self._form_group(group, factors, **kwargs)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Group [{self._name_tag}] containing {len(self._group)} models"

    def __len__(self):
        return len(self._group)

    def items(self):
        return self._group.items()

    def __iter__(self):
        for k, v in self.items():
            yield v

    def __contains__(self, item):
        return item in self._group

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._group[item]
        elif isinstance(item, list) or isinstance(item, np.ndarray):
            return self.__class__(group=[self._group[i] for i in item], **{attrn: {modn: attr for modn, attr in v.items()
                                                                             if modn in item}
                                                                     for attrn, v in
                                                                     self._get_converted_grp_annot().items()})

    def __setitem__(self, key, value):
        if isinstance(value, cobra.Model):
            self._group[key] = Model(model=value, name_tag=key)
        elif isinstance(value, GEMComposite):
            value._name_tag = key
            self._group[key] = value
        else:
            raise TypeError("Inputted value should be a dict, list, Model, or a cobra.model")

    @property
    def annotation(self) -> pd.DataFrame:
        return self._handle_annotation(self._group, replacing_annot=self._group_annotation)

    def add_annotation(self, added, store_in_model=False):
        self._ck_annotations(group=self._group, new_annot=added, store_in_model=store_in_model)

    def _get_converted_grp_annot(self):
        original_annot = {}
        for mod_name, annot in self._group_annotation.items():
            for key, val in annot.items():
                if key not in original_annot:
                    original_annot[key] = {mod_name: val}
                else:
                    original_annot[key].update({mod_name: val})
        return original_annot

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
        return list(reduce(set.union, [set(g.reaction_ids) for _, g in self._group.items()]))

    @property
    def metabolite_ids(self):
        """
        All the metabolites ID inside this object

        Returns
        -------
        met_ids: list of str
        """
        return list(reduce(set.union, [set(g.metabolite_ids) for _, g in self._group.items()]))

    @property
    def gene_ids(self):
        """
        All the genes ID inside this object

        Returns
        -------
        gene_ids: list of str
        """
        return list(reduce(set.union, [set(g.gene_ids) for _, g in self._group.items()]))

    @property
    def subsystems(self):
        subs = {}
        for _, g in self._group.items():
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
        gene_data = {name: m.gene_data for name, m in self._group.items()}
        return GeneData.aggregate(gene_data, prop="data")

    def get_RAS(self, data_name, method="mean"):
        gene_data = {name: m.gene_data for name, m in self._group.items()}
        return GeneData.aggregate(gene_data, prop="data")

    def _get_group_model(self, group_by):
        if group_by is None:
            return {"model": [m for m in self._group.keys()]}

        model_annot = self._handle_annotation(self._group, replacing_annot=self._group_annotation)
        gb = model_annot.groupby(group_by).apply(lambda x: list(x.index))
        return {i: row for i, row in gb.items()}

    def aggregate_models(self, group_by):
        if group_by is None:
            return {self.name_tag: self}

        model_annot = self._handle_annotation(self._group, replacing_annot=self._group_annotation)
        gb = model_annot.groupby(group_by).apply(lambda x: list(x.index))  # {'gp1': [mod_name_1, mod_name_2,...], ...}
        return {gp_name: self[mod_names].rename(gp_name)
                for gp_name, mod_names in gb.items()}

    def get_rxn_info(self,
                     models,
                     attrs,
                     drop_duplicates=True):
        if models is not None and models != "all":
            selected_models = self[models]
        else:
            selected_models = self
        rxn_info_dfs = [mod.get_rxn_info(attrs) for mod in selected_models]
        mg_info = pd.concat(rxn_info_dfs, axis=0)
        if drop_duplicates:
            mg_info = mg_info.drop_duplicates()
        return mg_info

    def rename(self, name_tag, inplace=False):
        if inplace:
            self._name_tag = name_tag
            return
        print(f"create a new group containing {self._group}")
        return self.__class__(group=self._group, name_tag=name_tag)

    def do_flux_analysis(self,
                         method: str,
                         aggregate_method: str = "concat",
                         solver: str = "gurobi",
                         group_by: str = None,
                         **kwargs):
        """
        Do flux analysis on the models contained in this group.

        Parameters
        ----------
        method: str
            Analysis performed on the models.
        aggregate_method: str
            Aggregation method performed on the flux result.
        solver: str
            Solver used to do the analysis.
        group_by: str
            Used to determine the groups for the aggregate_method.
        kwargs: dict
            Keyword arguments used in the model.do_flux_analysis()

        Returns
        -------
        flux_result: FluxAnalysis

        """
        results = []
        for name, c in self._group.items():
            result = c.do_flux_analysis(method=method, solver=solver, **kwargs)
            result.add_categorical(c.name_tag, col_name="model")
            if c.name_tag in self._group_annotation:
                for ftr_name, ftr in self._group_annotation[c.name_tag].items():
                    result.add_categorical(ftr,
                                           col_name=ftr_name)
            results.append(result)
        gp_annot = pd.DataFrame({group_by: self.annotation[group_by].unique()},
                                index=self.annotation[group_by].unique()) if group_by is not None else self.annotation

        agg_result = results[0].__class__.aggregate(results, method=aggregate_method,
                                                    log={"name": self.name_tag,
                                                         "group_by": group_by if group_by is not None else "model",
                                                         "group": self._get_group_model(group_by)},
                                                    group_annotation=gp_annot,
                                                    rxn_annotation=self.get_rxn_info(models="all", attrs=["subsystem"]))

        return agg_result

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

    @staticmethod
    def _handle_annotation(group, included_attrs="n_comp", replacing_annot=None) -> pd.DataFrame:
        annot = {m_name: m.annotation for m_name, m in group.items()}
        if included_attrs == "n_comp":
            included_attrs = ["n_rxns", "n_genes", "n_mets"]
        elif included_attrs is None:
            included_attrs = []
        elif not isinstance(included_attrs, list):
            raise TypeError(f"included_attrs should be either None, a list, or 'n_comp'. "
                            f"Got {type(included_attrs)} instead.")
        if replacing_annot is None:
            replacing_annot = {}

        for ia in included_attrs:
            for m_name, mod in group.items():
                annot[m_name].update({ia: getattr(mod, ia) if hasattr(mod, ia) else None})
        for m, a in replacing_annot.items():
            annot[m].update(a)

        return pd.DataFrame(annot).T

    @staticmethod
    def _form_group_chk_list(group_list, existing_gp=None):
        if not all([isinstance(c, GEMComposite) for c in group_list]):
            raise ValueError("The input list should be a list of pipeGEM.Model, the list contains types: ",
                             set([type(c) for c in group_list]))
        if not len(group_list) == len(set([c.name_tag for c in group_list])):
            raise ValueError("At least two models have non-unique name. "
                             "Please consider rename the input models or drop the models with the same names.")
        if existing_gp is not None:
            if any([c.name_tag in existing_gp for c in group_list]):
                raise ValueError(f"At least two models have non-unique name. "
                                 "Please consider rename the input models or drop the models with the same names.")

    def _ck_annotations(self, group, new_annot, factor_df=None, store_in_model=False):
        if factor_df is not None:
            factors = factor_df.to_dict(orient='dict')
            if not all([mn in group for key, mg in factors.items() for mn in mg]):
                raise KeyError("Some names in the factors are not in the model group",
                               f"{[mn for key, mg in factors.items() for mn in mg if mn not in group]}")
            for key, annot_dic in factors.items():
                for mod_name, val in annot_dic.items():
                    if store_in_model:
                        group[mod_name].add_annotation(key=key, value=val)
                    else:
                        self._group_annotation[mod_name][key] = val

        for key, annot_dic in new_annot.items():
            if all([isinstance(mod_names, list) or isinstance(mod_names, np.ndarray)
                    for val, mod_names in annot_dic.items()]):
                for val, mod_names in annot_dic.items():
                    if not all([mn in group for mn in mod_names]):
                        raise KeyError("Some names in the new annotation dict are not in the model group",
                                       f"{[mn for mn in annot_dic if mn not in group]}")

                for val, mod_names in annot_dic.items():
                    for mod_name in mod_names:
                        if store_in_model:
                            group[mod_name].add_annotation(key=key, value=val)
                        else:
                            self._group_annotation[mod_name][key] = val
            else:
                if not all([mn in group for mn in annot_dic]):
                    raise KeyError("Some names in the new annotation dict are not in the model group:",
                                   f"{[mn for mn in annot_dic if mn not in group]}")
                for mod_name, val in annot_dic.items():
                    if store_in_model:
                        group[mod_name].add_annotation(key=key, value=val)
                    else:
                        self._group_annotation[mod_name][key] = val

    def _form_group(self, group_dict, factors=None, store_in_model=False, **kwargs) -> dict:
        groups = {}
        if isinstance(group_dict, list) or isinstance(group_dict, np.ndarray):
            self._form_group_chk_list(group_dict)
            groups.update({c.name_tag: c for c in group_dict})
            for c_name, c in groups.items():
                self._group_annotation[c_name] = {}
        elif isinstance(group_dict, dict):
            for name, comp in group_dict.items():
                if isinstance(comp, cobra.Model):
                    groups[name] = Model(model=comp, name_tag=name)
                    self._group_annotation[name] = {}
                elif isinstance(comp, Model):
                    groups[name] = comp
                    self._group_annotation[name] = {}
                    if name != comp.name_tag:
                        warn(f"Assigned name ({name}) is not equal to model's name_tag ({comp.name_tag})")
                elif isinstance(comp, list):
                    self._form_group_chk_list(comp, existing_gp=groups)
                    for c in comp:
                        self._group_annotation[c.name_tag] = {"group_name": name}
                        groups[c.name_tag] = c
                elif isinstance(comp, dict):
                    pg_mod_comp = {c_name: Model(model=c, name_tag=c_name) for c_name, c in comp.items()}
                    self._form_group_chk_list([c for _, c in pg_mod_comp.items()], existing_gp=groups)
                    for c_name, c in pg_mod_comp.items():
                        self._group_annotation[c_name] = {"group_name": name}
                        groups[c_name] = c
        elif group_dict is None:
            groups = {}
        else:
            raise ValueError("Group doesn't support object type: ", type(group_dict))

        self._ck_annotations(group=groups,
                             new_annot=kwargs,
                             factor_df=factors,
                             store_in_model=store_in_model)
        return groups

    def get_info(self,
                 models=None,
                 features=None) -> pd.DataFrame:
        """
        Get a information table by traversing the object structure

        Parameters
        ----------
        models: optional, str
            The name tag of the selected models, if None, use all models.
        features: optional, list of str, str
            The features to be obtained while the traverse
        Returns
        -------
        information_table: pd.DataFrame

        """
        if models is None:
            group = self._group
        else:
            group = [m for n, m in self._group.items() if n in models]
        return self._handle_annotation(group,
                                       included_attrs=features,
                                       replacing_annot=self._group_annotation)

    def _compare_components_jaccard(self,
                                    models,
                                    group_by=None,
                                    components="all"):
        group_list = [g for g in models] if group_by is None else \
                [g for g in models.aggregate_models(group_by=group_by).values()]

        label_index = {grp.name_tag: ind
                       for ind, grp in enumerate(group_list)}
        jaccard_index = {f'{A.name_tag}_to_{B.name_tag}': calc_jaccard_index(A, B, components)
                         for A, B in itertools.combinations(group_list, 2)}
        jaccard_index.update({f'{A.name_tag}_to_{A.name_tag}': 1 for A in group_list})
        group_names = [A.name_tag for A in group_list]
        comp_arr = np.array([[jaccard_index[f'{sorted([A, B], key=lambda x: label_index[x])[0]}'
                                          f'_to_'
                                          f'{sorted([A, B], key=lambda x: label_index[x])[1]}']
                            for A in group_names]
                            for B in group_names])
        comp_arr = pd.DataFrame(comp_arr, index=group_names, columns=group_names)
        result = ComponentComparisonAnalysis(log={"components": components,
                                                  "group_by": group_by})
        result.add_result({"comparison_df": comp_arr,
                           "group_annotation": self.annotation if group_by is None else
                           self.annotation.groupby(group_by).apply(lambda x: list(x.index)).to_frame()
                           })
        return result

    def _compare_component_num(self,
                               models,
                               group_by=None,
                               components="all"):

        components = ['genes', 'reactions', 'metabolites'] if components == "all" else components
        c_to_f = {"genes": "n_genes", "reactions": "n_rxns", "metabolites": "n_mets"}
        comp_df = models.get_info(features=[c_to_f[c] for c in components]).reset_index().rename(columns={"index": "model"})
        id_vars = ["model"] + ([] if group_by is None else [group_by])
        new_comp_df = pd.concat(
            (pd.melt(comp_df, id_vars=id_vars, value_vars="n_rxns", var_name="component", value_name="number"),
             pd.melt(comp_df, id_vars=id_vars, value_vars="n_mets", var_name="component", value_name="number"),
             pd.melt(comp_df, id_vars=id_vars, value_vars="n_genes", var_name="component",
                     value_name="number")),
            ignore_index=True
        )

        new_comp_df["number"] = new_comp_df["number"].astype(dtype=int)
        result = ComponentNumberAnalysis(log={"components": components, 'group_by': group_by})
        result.add_result({"comp_df": new_comp_df})
        return result

    def _compare_component_PCA_helper(self, comp_id_dic, model, comp_name, name):
        if name not in comp_id_dic:
            comp_id_dic[name] = getattr(model, comp_name)
        else:
            comp_id_dic[name].extend(getattr(model, comp_name))

    def _compare_component_PCA(self,
                               models,
                               group_by=None,
                               components="all",
                               n_components=2,
                               incremental=False,
                               **kwargs) -> PCA_Analysis:
        components = ['gene_ids', 'reaction_ids', 'metabolite_ids'] if components == "all" else components
        comp_id_dic, comp_dfs, group_info = {}, {}, {}
        group_list = [g for g in models] if group_by is None else \
            [g for g in models.aggregate_models(group_by=group_by).values()]

        for c in components:
            for g in group_list:
                self._compare_component_PCA_helper(comp_id_dic, g, c, g.name_tag)

        for m_name, comp_ids in comp_id_dic.items():
            comp_dfs[m_name] = pd.Series(data=np.ones((len(comp_ids), )), index=comp_ids)
        component_df = pd.DataFrame(comp_dfs).fillna(0).T
        del comp_dfs
        pca_fitted_df, pca_expvar_df, pca_comp_df = prepare_PCA_dfs(component_df.T,
                                                                    n_components=n_components,
                                                                    incremental=incremental,
                                                                    **kwargs)
        result = PCA_Analysis(log={"group_name_tag": self.name_tag,
                                   "dr_method": "PCA"})
        result.add_result({"PC": pca_fitted_df,
                           "exp_var": pca_expvar_df,
                           "components": pca_comp_df,
                           "group_annotation": self.annotation if group_by is None else
                                self.annotation.groupby(group_by).apply(lambda x: list(x.index)).to_frame()})
        return result

    def compare(self,
                models=None,
                group_by="group_name",
                method: str = "jaccard",
                **kwargs
                ):
        assert method in ["jaccard", "PCA", "num"]

        models: Group = self[models] if models is not None else self
        if method == "jaccard":
            return self._compare_components_jaccard(models=models, group_by=group_by, **kwargs)
        elif method == "num":
            return self._compare_component_num(models=models, group_by=group_by, **kwargs)
        elif method == "PCA":
            return self._compare_component_PCA(models=models, group_by=group_by, **kwargs)  #TODO



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
