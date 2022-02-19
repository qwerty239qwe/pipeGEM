from statistics import mean, median
from warnings import warn
from typing import Dict, Optional, List

import json
import pandas as pd
import numpy as np
from pipeGEM.utils import get_subsystems


def identity_transform(x): return x


class Expression:
    def __init__(self, model, data, transform=None,
                 method='default',
                 expression_threshold=1e-4,
                 missing_value=np.nan,
                 convert_to_str=True,
                 load_mode=False,
                 loaded_rxn_scores=None,
                 loaded_rxn_score_in_subsystem=None,
                 loaded_subsystems=None):

        self.missing_value = missing_value
        if not load_mode:
            self.subsystems = get_subsystems(model)

            if not transform:
                transform = identity_transform

            if isinstance(data, pd.Series):
                self.data = {str(gene_id) if convert_to_str else gene_id:
                                 transform(exp) if exp > expression_threshold else 0 for gene_id, exp in
                             zip(data.index, data)}
            elif isinstance(data, dict):
                self.data = {str(gene_id) if convert_to_str else gene_id:
                                 transform(exp) if exp > expression_threshold else 0
                             for gene_id, exp in data.items()}
            else:
                raise ValueError("Expression data should be a dict or a pandas series")
            self.omic_genes = list(self.data.keys())
            self.cal_rxn_scores(model, method, expression_threshold)
            self._rxn_score_in_subsystem = self._map_to_subsystem(model=model)
        else:
            self.data = data
            self._rxn_scores = loaded_rxn_scores
            self._rxn_score_in_subsystem = loaded_rxn_score_in_subsystem
            self.omic_genes = list(self.data.keys())
            self.subsystems = loaded_subsystems

    def save(self, file_name):
        state = {"missing_value": self.missing_value,
                 "_rxn_scores": self._rxn_scores,
                 "_rxn_score_in_subsystem": self._rxn_score_in_subsystem,
                 "data": self.data,
                 "subsystems": self.subsystems}
        with open(file_name, "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, file_name):
        with open(file_name, "r+") as json_file:
            kwargs = json.load(json_file)
        return cls(load_mode=True,
                   model=None,
                   data=kwargs.get("data"),
                   missing_value=kwargs.get("missing_value"),
                   loaded_rxn_scores=kwargs.get("_rxn_scores"),
                   loaded_rxn_score_in_subsystem=kwargs.get("_rxn_score_in_subsystem"),
                   loaded_subsystems=kwargs.get("subsystems"))

    @property
    def rxn_scores(self) -> Dict[str, float]:
        return self._rxn_scores

    def cal_rxn_scores(self, model, method='default', expression_threshold=1e-4):
        self._rxn_scores = self._map_to_rxns(model, method, expression_threshold)

    @property
    def rxn_score_in_subsystem(self):
        return self._rxn_score_in_subsystem

    def get_rxn_score(self, rxn_id):
        return self._rxn_scores[rxn_id]

    def _inner_grr_helper(self, grr_list) -> list:
        inner_grr_scores = []
        for g in grr_list.split('and'):
            if g == "":
                inner_grr_scores.append(-1)

            elif (g[:g.index(".")] if "." in g else g) in self.omic_genes:
                inner_grr_scores.append(self.data[g[:g.index(".")] if "." in g else g])
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

    def _map_to_rxns(self, model, method='default', threshold=0.):
        """
        :return: dict {rxn_id : score}
        """
        grrs = {r.id: r.gene_reaction_rule.replace(' ', '').replace('(', '').replace(')', '')
                for r in model.reactions}

        rxn_score = dict()
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
            if method == 'default' and v <= threshold:
                rxn_score[k] = 0
            elif method == 'mCADRE' and v <= threshold:
                rxn_score[k] = -1
        return rxn_score

    def _map_to_subsystem(self, model):
        rxn_score_in_subsystem = {subsystem: [] for subsystem in self.subsystems}
        for rxn in model.reactions:
            rxn_score_in_subsystem[rxn.subsystem].append(self._rxn_scores[rxn.id])
        return rxn_score_in_subsystem

    def get_subsystem_scores(self, subsystems='all', statistics='mean') -> dict:

        method_dict = {'total': sum, 'mean': mean, 'median': median}

        if subsystems == 'all':
            subsystems = self.subsystems

        if statistics not in ['total', 'mean', 'median']:
            raise ValueError('Choose method in (total, mean, median)')

        if not self._rxn_scores:
            raise ValueError("rxn_scores is somewhat missing. Please recalculate it using cal_rxn_scores")

        subsystem_score = {}
        for subsystem, data in self.rxn_score_in_subsystem.items():
            if subsystem in subsystems:
                not_na_data = [d for d in data if not np.isnan(d)]
                if len(not_na_data) == 0:
                    subsystem_score[subsystem] = 0
                else:
                    subsystem_score[subsystem] = method_dict[statistics](not_na_data)
        not_exist_subsystems = [err for err in subsystems if err not in self.subsystems]
        if not_exist_subsystems:
            warn(f"{not_exist_subsystems} are not model's subsystems!")

        return {subsystem: data for subsystem, data in subsystem_score.items() if subsystem not in not_exist_subsystems}


def map_data_to_rxns(data_df,
                     ref_model,
                     sample_names: Optional[List[str]] = None,
                     missing_value: float = 0,
                     expression_threshold: float = 1e-4):
    expression_dic = {}
    for sample in data_df.columns:
        if sample_names is None or sample in sample_names:
            expression_dic[sample] = Expression(ref_model,
                                                data_df[sample],
                                                missing_value=missing_value,
                                                expression_threshold=expression_threshold)
    return expression_dic

