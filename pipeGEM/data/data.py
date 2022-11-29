from typing import Dict, Optional, List
from pathlib import Path

import numpy as np
import pandas as pd
from pint import UnitRegistry

from ._base import BaseData
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


class RxnMapper:
    def __init__(self,
                 data,
                 model,
                 threshold=0,
                 absent_value=0,
                 missing_value=np.nan,
                 and_operation="nanmin",
                 or_operation="nanmax",
                 plus_operation="nansum"  # for reduced reactions
                 ):
        self.genes = data.genes
        self.gene_data = data.gene_data
        self.missing_value = missing_value  # if the gene is not shown in the given data
        self.rxn_scores = self._map_to_rxns(model,
                                            threshold=threshold,
                                            absent_value=absent_value,
                                            and_operation=and_operation,
                                            or_operation=or_operation,
                                            plus_operation=plus_operation)

    def _inner_grr_helper(self, grr_list) -> list:
        inner_grr_scores = []
        for g in grr_list.split('and'):
            if g == "" or not (g[:g.index(".")] if "." in g else g) in self.genes:
                inner_grr_scores.append(self.missing_value)
            else:
                inner_grr_scores.append(self.gene_data[g[:g.index(".")] if "." in g else g])
        return inner_grr_scores

    def _outer_grr_helper(self, inner_grr_scores, operation="nanmin") -> float:
        if len(inner_grr_scores) == 0:
            return self.missing_value
        if all([not np.isfinite(i) for i in inner_grr_scores]):
            return self.missing_value
        return getattr(np, operation)(inner_grr_scores)

    def _map_to_rxns(self,
                     model,
                     absent_value=0,
                     threshold=0.,
                     and_operation="nanmin",
                     or_operation="nanmax",
                     plus_operation="nansum"):
        """
        :return: dict {rxn_id : score}
        """
        grrs = {r.id: r.gene_reaction_rule.replace(' ', '').replace('(', '').replace(')', '')
                for r in model.reactions}

        rxn_score = {}
        for rxn_id, grr in grrs.items():
            if len(grr) == 0:
                rxn_score[rxn_id] = self.missing_value
                continue
            plus_grr_list = []
            for grr_i in grr.split("plus"):
                outer_grr_list = []
                for gl in grr_i.split("or"):
                    inner_grr_scores = self._inner_grr_helper(gl)
                    outer_grr_list.append(self._outer_grr_helper(inner_grr_scores, and_operation))
                plus_grr_list.append(self._outer_grr_helper(outer_grr_list, or_operation))
            if all([not np.isfinite(i) for i in plus_grr_list]):
                rxn_score[rxn_id] = self.missing_value
            else:
                rxn_score[rxn_id] = getattr(np, plus_operation)(plus_grr_list)
            rxn_score[rxn_id] = rxn_score[rxn_id] if np.isfinite(rxn_score[rxn_id]) else self.missing_value

        for k, v in rxn_score.items():
            if v <= threshold:
                rxn_score[k] = absent_value
        return rxn_score


class GeneData(BaseData):
    def __init__(self,
                 data,
                 convert_to_str: bool = True,
                 expression_threshold: float = 1e-4,
                 absent_expression: float = 0,
                 data_transform = None,
                 discrete_transform = None,
                 ordered_thresholds: list = None):
        """
        A GeneData object stores gene data using a dict. It can also calculate rxn scores for a given model.

        Parameters
        ----------
        data: pd.Series or a dict
        convert_to_str: bool
            Convert the gene names into strings
        expression_threshold: float
        absent_expression: float
        data_transform
        discrete_transform
        ordered_thresholds: list
            Ascending thresholds indicating how the gene level will be transformed discretely

        Examples
        ----------
        """
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
        else:
            raise ValueError("Expression data should be a dict or a pandas series")
        self.genes = list(self.gene_data.keys())
        self._digitize_data(ordered_thresholds)
        self.rxn_mapper = None

    def __getitem__(self, item):
        return self.gene_data[item]

    def align(self, model, **kwargs):
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

    def get_threshold(self, name, **kwargs):
        tf = threshold_finders.create(name)
        return tf.find_threshold(self.gene_data, **kwargs)

    def assign_local_threshold(self, local_threshold_result, method="binary", group=None, **kwargs):
        assert method in ["binary", "ratio", "diff", "rdiff"]
        group = group if group is not None else 'exp_th'
        gene_exp_ths = local_threshold_result.exp_ths[group]
        data_and_ths = pd.concat([gene_exp_ths, pd.DataFrame({"data": self.gene_data})], axis=1)
        if method == "binary":
            self.gene_data = (data_and_ths["data"] > data_and_ths[group]).astype(int).to_dict()
        elif method == "ratio":
            self.gene_data = (data_and_ths["data"] / data_and_ths[group]).to_dict()
        elif method == "diff":
            self.gene_data = (data_and_ths[group] - data_and_ths["data"]).to_dict()
        elif method == "rdiff":
            self.gene_data = (data_and_ths["data"] - data_and_ths[group]).to_dict()

    @classmethod
    def aggregate(cls, data, method="concat", prop="data", absent_expression=0) -> DataAggregation:
        assert prop in ["data", "score"], "prop should be either data or score"
        assert all([isinstance(v, dict) for k, v in data.items()]) or all([isinstance(v, GeneData) for k, v in data.items()])

        obj_prop = {"data": "gene_data", "score": "rxn_scores"}

        if all([isinstance(v, dict) for k, v in data.items()]):
            mg_d = pd.concat([pd.DataFrame({name+":"+d_name: getattr(gene_data, obj_prop[prop])})
                              for name, d in data.items() for d_name, gene_data in d.items()], axis=1).fillna(absent_expression)
        else:
            mg_d = pd.concat([pd.DataFrame({name: getattr(gene_data, obj_prop[prop])})
                              for name, gene_data in data.items()], axis=1).fillna(absent_expression)
        if method != "concat":
            mg_d = getattr(mg_d, method)(axis=1).to_frame()
        result = DataAggregation(log={"method": method, "prop": prop, "absent_expression": absent_expression})
        result.add_result(mg_d)
        return result


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
    def from_file(cls, file_name="DMEM", **kwargs):
        medium_file_dir = Path(__file__).parent.parent.parent / "medium"
        if (medium_file_dir / file_name).with_suffix(".tsv").is_file():
            data = pd.read_csv((medium_file_dir / file_name).with_suffix(".tsv"), sep='\t', index_col=0)
        else:
            data = pd.read_csv(file_name, sep='\t', index_col=0)
        return cls(data, **kwargs)

