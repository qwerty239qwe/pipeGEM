from ._base import *
import json
import pandas as pd
from ast import literal_eval


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
        self._result_loading_params = {"result_df": dict(converters = {'Obj_rxns': lambda x: literal_eval(x) if pd.notna(x) and x != "" else x,
                                                                       'task_support_rxns': lambda x: literal_eval(x) if pd.notna(x) and x != "" else x,
                                                                       'task_support_rxn_fluxes': lambda x: literal_eval(x) if pd.notna(x) and x != "" else x,
                                                                       'Sink Status': lambda x: literal_eval(x) if pd.notna(x) and x != "" else x,
                                                                       'rxn_supps': lambda x: literal_eval(x) if pd.notna(x) and x != "" else x})}

    @property
    def model_score(self):
        return self._result["score"]

    @property
    def task_support_rxns(self) -> dict:
        return dict(self._result["result_df"]["task_support_rxns"].items())

    @property
    def task_support_rxn_fluxes(self) -> dict:
        return dict(self._result["result_df"]["task_support_rxn_fluxes"].items())

    @property
    def rxn_supps(self) -> dict:
        return dict(self._result["result_df"]["rxn_supps"].items())

    @property
    def task_ids(self) -> list:
        return self._result["result_df"].index.to_list()

    def get_task_support_rxns(self,
                              task_id,
                              include_supps=True):
        return self.task_support_rxns[task_id][0] + (self.rxn_supps[task_id][0]
                                                     if include_supps and
                                                        task_id in self.rxn_supps and
                                                        isinstance(self.rxn_supps[task_id][0], list)
                                                     else [])

    def get_all_possible_sups(self, task_id):
        return self.task_support_rxns[task_id]

    def is_essential(self, task_id, rxn_id) -> bool:
        return all([rxn_id in supp_path for supp_path in self.task_support_rxns[task_id]])