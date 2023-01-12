from ._base import *
import json
import pandas as pd


class TaskAnalysis(BaseAnalysis):
    result_df_default_colnames = ['Passed', 'Should fail',
                                  'Missing mets', 'Status',
                                  'Obj_value', 'Obj_rxns',
                                  'Sink Status']

    def __init__(self, log):
        self._result_df = None
        self._model_score = 0
        self._task_support_rxns = {}
        self._task_support_rxn_fluxes = {}

        self._rxn_supps = {}
        super(TaskAnalysis, self).__init__(log)

    @classmethod
    def load(cls, file_name, **kwargs):

        with open(file_name) as json_file:
            data = json.load(json_file)
            task_analysis = cls(log=data.pop("log"))
            result_dic = {k: {task_id: task_results[k] for task_id, task_results in data.items()}
                          for k in cls.result_df_default_colnames}

            for task_id, task_results in data.items():
                tr, trf, rs = task_results.get("task_support_rxns"), \
                              task_results.get("task_support_rxn_fluxes"), \
                              task_results.get("rxn_supps")
                if tr is not None:
                    task_analysis._task_support_rxns[task_id] = tr
                if trf is not None:
                    task_analysis._task_support_rxn_fluxes[task_id] = trf
                if rs is not None:
                    task_analysis._rxn_supps[task_id] = rs
            result_dic.update({"task_support_rxns": task_analysis._task_support_rxns,
                               "task_support_rxn_fluxes": task_analysis._task_support_rxn_fluxes,
                               "rxn_supps": task_analysis._rxn_supps})

            task_analysis._result_df = pd.DataFrame(result_dic)

        return task_analysis

    def save(self, file_name):
        result_dic = {}
        for c in self.result_df_default_colnames:
            for k, v in self.result_df[c].to_dict().items():
                if k not in result_dic:
                    result_dic[k] = {}
                if c != "Obj_rxns":
                    result_dic[k][c] = v if not pd.isna(v) else None
                else:
                    result_dic[k][c] = [vi.id for vi in v]

        for attr_name, attr in zip(["task_support_rxns", "task_support_rxn_fluxes", "rxn_supps"],
                                   [self._task_support_rxns, self._task_support_rxn_fluxes, self._rxn_supps]):
            for k, v in attr.items():
                if not(isinstance(v, list) or isinstance(v, dict)) and pd.isna(v):
                    result_dic[k][attr_name] = None
                else:
                    result_dic[k][attr_name] = v
        result_dic["log"] = self._log
        with open(file_name, "w") as f:
            json.dump(result_dic, f)

    @property
    def model_score(self):
        return self._model_score

    @property
    def task_support_rxns(self):
        return self._task_support_rxns

    @property
    def result(self):
        return self._result_df

    @property
    def result_df(self):
        return self._result_df

    def add_result(self, result_df, score):
        self._result_df = result_df
        self._model_score = score
        self._task_support_rxns = dict(self._result_df["task_support_rxns"].items())
        self._task_support_rxn_fluxes = dict(self._result_df["task_support_rxn_fluxes"].items())
        self._rxn_supps = dict(self._result_df["rxn_supps"].items())

    def get_task_support_rxns(self, task_id, include_supps=True):
        return self._task_support_rxns[task_id] + (self._rxn_supps[task_id]
                                                   if include_supps and task_id in self._rxn_supps and isinstance(self._rxn_supps[task_id], list)
                                                   else [])

