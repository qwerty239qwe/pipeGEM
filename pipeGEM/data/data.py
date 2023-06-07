from typing import Dict, Union
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from pint import UnitRegistry

from ._base import BaseData
from pipeGEM.analysis import RxnMapper
from pipeGEM.analysis import threshold_finders
from pipeGEM.analysis import DataAggregation

HPA_scores = {'High': 20,
              'Medium': 15,
              'Low': 10,
              'None': -8,
              'Strong': 20,
              'Moderate': 15,
              'Weak': 10,
              'Negative': -8,
              'Not detected': -8}

dis_trans = {"HPA": HPA_scores}


class GeneData(BaseData):
    """
    A GeneData object stores gene data using a dict. It can also calculate rxn scores for a given model.

    Parameters
    ----------
    data: pd.Series, ad.AnnData, or dict
        Data contains pairs of gene IDs and expression values.
    convert_to_str: bool
        Convert the gene names into strings
    expression_threshold: float
        The absent_expression will be assigned to the expression values below this threshold
    absent_expression: float
        The value assigned to the low-expressed genes
    data_transform: callable
        Transformation applied to the rxn_scores and the transformed_gene_data. e.g. np.log2
    discrete_transform: str, dict, or callable
        Discrete data transformation applied to the rxn_scores.
    ordered_thresholds: list
        Ascending thresholds indicating how the gene level will be transformed discretely

    Examples
    ----------
    """

    def __init__(self,
                 data: Union[ad.AnnData, pd.Series, dict],
                 convert_to_str: bool = True,
                 expression_threshold: float = 1e-4,
                 absent_expression: float = 0,
                 data_transform = None,
                 discrete_transform = None,
                 ordered_thresholds: list = None):

        super().__init__("genes")
        self.convert_to_str = convert_to_str
        discrete_transform = self._parse_discrete_transform(discrete_transform, ordered_thresholds)
        self.data_transform = (lambda x: x) if data_transform is None else data_transform

        if isinstance(data, pd.Series):
            data = data.copy().apply(lambda x: discrete_transform(x)) if discrete_transform is not None else data.copy()
            self.gene_data = {str(gene_id) if convert_to_str else gene_id:
                              exp if exp > expression_threshold else absent_expression
                              for gene_id, exp in zip(data.index, data)}
        elif isinstance(data, dict):
            data = {k: discrete_transform(v) if discrete_transform is not None else v for k, v in data.items()}
            self.gene_data = {str(gene_id) if convert_to_str else gene_id:
                              exp if exp > expression_threshold else absent_expression
                              for gene_id, exp in data.items()}
        elif isinstance(data, ad.AnnData):
            if data.obs.shape[0] > 1:
                raise ValueError(f"Input anndata should contain only one observation, found {data.obs.shape[0]} instead.")
            self.gene_data = {str(gene_id) if convert_to_str else gene_id:
                              exp if exp > expression_threshold else absent_expression
                              for gene_id, exp in zip(data.var.index, data[0, :].X.toarray().ravel())}
        else:
            raise ValueError("Expression data should be a dict, an AnnData or a pandas series.")
        self.genes = list(self.gene_data.keys())
        self._digitize_data(ordered_thresholds)
        self.rxn_mapper = None

    def __getitem__(self, item):
        return self.gene_data[item]

    def align(self, model, **kwargs):
        """
        Calculate rxn_scores using a metabolic model.

        Parameters
        ----------
        model: pipeGEM.Model or cobra.Model
            The model with the genes and reactions to be mapped onto
        kwargs:
            Keyword arguments used to create a RxnMapper object, including:
            threshold: float or int, default = 0
                The absent_value will be assigned to the expression values below this threshold
            absent_value: float or int, default = 0
                The value assigned to the low-expressed genes
            missing_value: any, default = np.nan
                The value assigned to the genes not included in the gene_data
            and_operation: str, default = 'nanmin',
                The operation name used to calculate the 'and' gene-reaction relationships.

                Valid operations include:
                nanmin: return minimum while ignoring all the nan values
                nanmax: return maximum while ignoring all the nan values
                nansum: return the expression sums while ignoring all the nan values
                nanmean: calculate the expression means while ignoring all the nan values
            or_operation: str, default = 'nanmax'
                The operation name used to calculate the 'or' gene-reaction relationships.
            plus_operation: str, default = 'nansum'
                The operation name used to calculate the 'plus' gene-reaction relationships.

        Returns
        -------
        None
        """
        self.rxn_mapper = RxnMapper(self, model, **kwargs)

    def transformed_rxn_scores(self, func):
        return {k: func(v) for k, v in self.rxn_scores.items()}

    @property
    def transformed_gene_data(self):
        return {k: self.data_transform(v) for k, v in self.gene_data.items()}

    @property
    def rxn_scores(self) -> Dict[str, float]:
        if self.rxn_mapper is None:
            raise AttributeError("The rxn mapper is not initialized. Please do .align(model) first.")
        return {k: self.data_transform(v) for k, v in self.rxn_mapper.rxn_scores.items()}

    def _digitize_data(self, ordered_thresholds):
        if ordered_thresholds is not None:
            n_bins = len(ordered_thresholds)
            ranges = np.array([i for i in range(-((n_bins - 1) // 2), n_bins // 2 + 1)])
            data = np.array(list(self.gene_data.values()))
            disc_data = ranges[np.digitize(data, ordered_thresholds)]
            self.gene_data = dict(zip(self.gene_data.keys(), disc_data))

    @staticmethod
    def _parse_discrete_transform(discrete_transform, ordered_thresholds):
        if isinstance(discrete_transform, str):
            if discrete_transform in dis_trans:
                return lambda x: dis_trans[discrete_transform][x]
        if isinstance(discrete_transform, dict):
            return lambda x: discrete_transform[x]
        if callable(discrete_transform) or discrete_transform is None:
            return discrete_transform
        raise ValueError("Discrete transform is not a valid value, choose a dict or a str as input")

    def calc_rxn_score_stat(self,
                            rxn_ids,
                            ignore_na=True,
                            na_value=0,
                            return_if_all_na=-1,
                            method="mean"):
        scores = np.array([v for k, v in self.rxn_scores.items() if k in rxn_ids])
        if all([np.isnan(i) for i in scores]):
            return return_if_all_na

        if not ignore_na:
            scores[np.isnan(scores)] = na_value
        if method == "mean":
            return np.nanmean(scores)
        if method == "median":
            return np.nanmedian(scores)

        raise ValueError("The method is not supported")

    def apply(self, func):
        return {k: func(v) for k, v in self.rxn_mapper.rxn_scores.items()}

    def get_threshold(self, name, transform=True, **kwargs):
        tf = threshold_finders.create(name)
        return tf.find_threshold({gn: self.data_transform(gv) if transform else gv
                                  for gn, gv in self.gene_data.items()}, **kwargs)

    def assign_local_threshold(self, local_threshold_result, transform=True, method="binary", group=None, **kwargs):
        assert method in ["binary", "ratio", "diff", "rdiff"]
        group = group if group is not None else 'exp_th'
        gene_exp_ths = local_threshold_result.exp_ths[group]
        data_and_ths = pd.concat([gene_exp_ths, pd.DataFrame({"data": {gn: self.data_transform(gv) if transform else gv
                                                                       for gn, gv in self.gene_data.items()}})], axis=1)
        if method == "binary":
            self.gene_data = (data_and_ths["data"] > data_and_ths[group]).astype(int).to_dict()
        elif method == "ratio":
            self.gene_data = (data_and_ths["data"] / data_and_ths[group]).to_dict()
        elif method == "diff":
            self.gene_data = (data_and_ths[group] - data_and_ths["data"]).to_dict()
        elif method == "rdiff":
            self.gene_data = (data_and_ths["data"] - data_and_ths[group]).to_dict()

    @classmethod
    def aggregate(cls,
                  data,
                  method="concat",
                  prop="data",
                  absent_expression=0) -> DataAggregation:
        assert prop in ["data", "score"], "prop should be either data or score"

        obj_prop = {"data": "gene_data", "score": "rxn_scores"}

        if all([isinstance(v, dict) for k, v in data.items()]):
            mg_d = pd.concat([pd.DataFrame({name+":"+d_name: getattr(gene_data, obj_prop[prop])})
                              for name, d in data.items() for d_name, gene_data in d.items()], axis=1).fillna(absent_expression)
        else:
            mg_d = pd.concat([pd.DataFrame({name: getattr(gene_data, obj_prop[prop])})
                              for name, gene_data in data.items()], axis=1).fillna(absent_expression)
        if method != "concat":
            mg_d = getattr(mg_d, method)(axis=1).to_frame()
        result = DataAggregation(log={"method": method,
                                      "prop": prop,
                                      "absent_expression": absent_expression,
                                      "group": _data_parse_group_models(data)})
        result.add_result(dict(agg_data=mg_d))
        return result


def _data_parse_group_models(data_group) -> dict:
    group_struct = {}
    for k, v in data_group.items():
        group_struct[k] = []
        if isinstance(v, dict):
            for vk, vv in v.items():
                group_struct[k].append(vk)
        else:
            group_struct[k].append(k)
    return group_struct


def find_local_threshold(data_df, **kwargs):
    tf = threshold_finders.create("local")
    return tf.find_threshold(data_df, **kwargs)


class ThermalData(BaseData):
    def __init__(self):
        super().__init__("metabolites")


class MediumData(BaseData):
    def __init__(self,
                 data,
                 conc_col_label="mmol/L",
                 conc_unit="mmol/L",
                 id_index=False,
                 name_index=True,
                 id_col_label="human_1",
                 name_col_label=None
                 ):
        super().__init__("metabolites")
        self.data_dict = dict(zip(data[id_col_label] if not id_index else data.index, data[conc_col_label]))
        self.rxn_dict = {}
        self.name_dict = dict(zip(data[id_col_label] if not id_index else data.index,
                                  data[name_col_label] if not name_index else data.index))
        self._u = UnitRegistry()
        self.conc_unit = self._u.Quantity(conc_unit)

    @staticmethod
    def _find_simple_rxn(rxns):
        cur_simp_ix, best_c, best_nm = -1, 1000, 1000
        for i, r in enumerate(rxns):
            nm = len(r.metabolites)
            c = sum([abs(c) for m, c in r.metabolites.items()])
            if best_nm > nm or (best_nm == nm and best_c > c):
                cur_simp_ix = i
                best_nm = nm
                best_c = c
        return rxns[cur_simp_ix]

    def align(self,
              model,
              external_comp_name="e",
              met_id_format="{met_id}{comp}"):
        exs = set([r.id for r in model.exchanges])

        for mid, conc in self.data_dict.items():
            m = model.metabolites.get_by_id(met_id_format.format(met_id=mid, comp=external_comp_name))
            m_related_r = set([r.id for r in m.reactions]) & exs
            if len(m_related_r) == 1:
                self.rxn_dict[list(m_related_r)[0]] = conc
            else:
                self.rxn_dict[self._find_simple_rxn(m.reactions).id] = conc

    def apply(self,
              model,
              cell_dgw=1e-12,
              n_cells_per_l = 1e6,
              time_hr=96,
              flux_unit="mmol/g/hr",
              threshold=1e-6
              ):
        cell_dgw = cell_dgw * self._u.gram
        n_cells_per_l = n_cells_per_l / self._u.liter
        time_hr = time_hr * self._u.hour
        for k in model.exchanges + model.sinks + model.demands:
            if k.id not in self.rxn_dict:
                if all(["C" not in m.formula for m in k.metabolites]):
                    print(k.id, " is an inorganic exchange reaction (not constrained)")
                elif all([c < 0 for m, c in k.metabolites.items()]):
                    k.lower_bound = 0
                elif all([c > 0 for m, c in k.metabolites.items()]):
                    k.upper_bound = 0
                else:
                    print(k.id, " is ignored")

        for r_id, conc in self.rxn_dict.items():
            influx_ub = (conc * self.conc_unit / (n_cells_per_l * cell_dgw) / time_hr).to(flux_unit).magnitude
            print(r_id, influx_ub)
            r = model.reactions.get_by_id(r_id)
            total_c = sum([c for m, c in r.metabolites.items()])
            if total_c < 0:
                r.bounds = (-max(influx_ub, threshold), r.bounds[1])
            else:
                r.bounds = (r.bounds[0], max(influx_ub, threshold))

    @classmethod
    def from_file(cls, file_name="DMEM", csv_kw=None, **kwargs):
        medium_file_dir = Path(__file__).parent.parent.parent / "medium"
        if (medium_file_dir / file_name).with_suffix(".tsv").is_file():
            data = pd.read_csv((medium_file_dir / file_name).with_suffix(".tsv"), sep='\t', index_col=0)
        else:
            csv_kw = csv_kw or {}
            data = pd.read_csv(file_name, **csv_kw)
        return cls(data, **kwargs)

