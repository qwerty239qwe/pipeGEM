from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from ._base import BaseData


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
                 missing_value=np.nan):
        self.genes = data.genes
        self.gene_data = data.gene_data
        self.missing_value = missing_value
        self.rxn_scores = self._map_to_rxns(model,
                                            threshold=threshold,
                                            absent_value=absent_value)

    def _inner_grr_helper(self, grr_list) -> list:
        inner_grr_scores = []
        for g in grr_list.split('and'):
            if g == "":
                inner_grr_scores.append(-1)

            elif (g[:g.index(".")] if "." in g else g) in self.genes:
                inner_grr_scores.append(self.gene_data[g[:g.index(".")] if "." in g else g])
            else:
                inner_grr_scores.append(np.inf)
        return inner_grr_scores

    @staticmethod
    def _outer_grr_helper(inner_grr_scores) -> float:
        if len(inner_grr_scores) == 0:
            return -1.
        if len(inner_grr_scores) == 1 and inner_grr_scores[0] == np.inf:
            return -np.inf
        elif all([i == np.inf for i in inner_grr_scores]):
            return -np.inf
        return min(inner_grr_scores)

    def _map_to_rxns(self,
                     model,
                     absent_value=0,
                     threshold=0.):
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
            outer_grr_list = []
            for gl in grr.split("or"):
                inner_grr_scores = self._inner_grr_helper(gl)
                outer_grr_list.append(self._outer_grr_helper(inner_grr_scores))
            rxn_score[rxn_id] = max(outer_grr_list)
            rxn_score[rxn_id] = rxn_score[rxn_id] if rxn_score[rxn_id] != -np.inf else self.missing_value

        for k, v in rxn_score.items():
            if v <= threshold:
                rxn_score[k] = absent_value
        return rxn_score


class GeneData(BaseData):
    def __init__(self,
                 data,
                 convert_to_str: bool = True,
                 expression_threshold = 1e-4,
                 absent_expression = 0,
                 data_transform = None,
                 discrete_transform = None):
        """
        A GeneData object stores gene data using a dict. It can also calculate rxn scores for a given model.

        Parameters
        ----------
        data
        convert_to_str
        expression_threshold
        data_transform
        discrete_transform


        Examples
        ----------
        data = GeneData(df)
        data.align(model)
        data.rxn_scores
        model.data["data1"] = data
        model.list_data
        """
        super().__init__("genes")
        self.convert_to_str = convert_to_str
        discrete_transform = self._parse_discrete_transform(discrete_transform)
        data_transform = lambda x: x if data_transform is None else data_transform

        if isinstance(data, pd.Series):
            data = data.copy().apply(lambda x: discrete_transform[x])
            self.gene_data = {str(gene_id) if convert_to_str else gene_id:
                              data_transform(exp) if exp > expression_threshold else absent_expression
                              for gene_id, exp in zip(data.index, data)}
        elif isinstance(data, dict):
            data = {k: discrete_transform[v] for k, v in data.items()}
            self.gene_data = {str(gene_id) if convert_to_str else gene_id:
                              data_transform(exp) if exp > expression_threshold else absent_expression
                              for gene_id, exp in data.items()}
        else:
            raise ValueError("Expression data should be a dict or a pandas series")
        self.genes = list(self.gene_data.keys())
        self.rxn_mapper = None

    def __getitem__(self, item):
        return self.gene_data[item]

    def align(self, model):
        self.rxn_mapper = RxnMapper(self, model, )

    @property
    def rxn_scores(self) -> Dict[str, float]:
        if self.rxn_mapper is None:
            raise AttributeError("The rxn mapper is not initialized. Please use .align(model) to do that first.")
        return self.rxn_mapper.rxn_scores

    @staticmethod
    def _parse_discrete_transform(discrete_transform):
        if isinstance(discrete_transform, str):
            if discrete_transform in dis_trans:
                return dis_trans[discrete_transform]
        if isinstance(discrete_transform, dict) or discrete_transform is None:
            return discrete_transform
        raise ValueError("Discrete transform is not a valid value, choose a dict or a str as input")


class ThermalData(BaseData):
    def __init__(self):
        super().__init__("metabolites")