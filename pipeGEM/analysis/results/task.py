from ._base import *
import json
import pandas as pd


class TaskAnalysis(BaseAnalysis):
    result_df_default_colnames = ['Passed', 'Should fail',
                                  'Missing mets', 'Status',
                                  'Obj_value', 'Obj_rxns',
                                  'Sink Status']

    def __init__(self, log):
        """
        An object containing task analysis result.
        This should contain results including:
            result_df: dict
                A dataframe recording the details of all tests
            score: int
                Number of passed functionality (metabolic task) tests.
        Parameters
        ----------
        log: dict
            A dict storing parameters used to perform this analysis
        """
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
        return self._result["score"]

    @property
    def task_support_rxns(self) -> dict:
        return dict(self._result_df["task_support_rxns"].items())

    @property
    def task_support_rxn_fluxes(self) -> dict:
        return dict(self._result_df["task_support_rxn_fluxes"].items())

    @property
    def rxn_supps(self) -> dict:
        return dict(self._result_df["rxn_supps"].items())

    @property
    def task_ids(self) -> list:
        return self._result["result_df"].index.to_list()

    def get_task_support_rxns(self,
                              task_id,
                              include_supps=True):
        return self.task_support_rxns[task_id][0] + (self.rxn_supps[task_id]
                                                     if include_supps and
                                                        task_id in self.rxn_supps and
                                                        isinstance(self.rxn_supps[task_id], list)
                                                     else [])

    def get_all_possible_sups(self, task_id):
        return self.task_support_rxns[task_id]

    def is_essential(self, task_id, rxn_id) -> bool:
        return all([rxn_id in supp_path for supp_path in self.task_support_rxns[task_id]])