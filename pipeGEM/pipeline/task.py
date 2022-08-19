from typing import List, Dict, Optional

import pandas as pd
import cobra
import numpy as np

from ._base import Pipeline
from .threshold import BimodalThreshold
from ..integration.mapping import Expression
from pipeGEM.analysis.tasks import TaskHandler, TASKS_FILE_PATH


class ReactionTester(Pipeline):
    def __init__(self,
                 task_tester: Optional[TaskHandler] = None,
                 task_file_path=TASKS_FILE_PATH,
                 model_compartment_format="[{}]",
                 constr_name = "default",
                 solver="gurobi"):
        super().__init__()
        self.constr_name: str = constr_name
        self.task_file_path = task_file_path
        self.model_compartment_format = model_compartment_format
        self.model_tester = task_tester
        self.solver = solver

    def run(self,
            expression_threshold: float,
            non_expression_threshold: float,
            rxn_scores: dict,
            ref_model: Optional[cobra.Model] = None,
            reset_tester: bool = False,
            test_sink: bool = True,
            **kwargs) -> (List[str], Dict[str, float]):
        if self.model_tester is None or reset_tester:
            self.model_tester = TaskHandler(ref_model,
                                           model_compartment_parenthesis=self.model_compartment_format,
                                           **kwargs)
            self.model_tester.test_all(test_sink=test_sink)
        self.model_tester.update_thresholds(express_thres=expression_threshold,
                                            non_express_thres=non_expression_threshold,
                                            rxn_scores=rxn_scores)
        self.model_tester.map_expr_to_tasks()
        return self.model_tester.get_all_protected_rxns(), self.model_tester.tasks_scores

# add validator


class TaskScoringPipeLine(Pipeline):
    def __init__(self,
                 saved_dist_plot_format = None,
                 use_first_guess = True,
                 task_tester: Optional[TaskHandler] = None,
                 task_file_path=TASKS_FILE_PATH,
                 model_compartment_format="[{}]",
                 constr_name = "default",
                 solver="gurobi"):
        super().__init__()
        self.rxn_tester = ReactionTester(task_tester=task_tester,
                                         task_file_path=task_file_path,
                                         model_compartment_format=model_compartment_format,
                                         constr_name=constr_name,
                                         solver=solver)
        self.threshold = BimodalThreshold(naming_format=saved_dist_plot_format,
                                          use_first_guess=use_first_guess)

    def run(self,
            model_dict: dict,
            data,
            rxn_score_trans=np.log2,
            test_sink=False,
            *args, **kwargs):
        expr_tol_dict, nexpr_tol_dict = {}, {}
        data = rxn_score_trans(data.copy())
        self.output = {}

        for sample in data.columns:
            expr_tol_dict[sample], nexpr_tol_dict[sample] = self.threshold(data=data[sample],
                                                                           sample_name=sample)
            rxn_scores = Expression(model_dict[sample], data[sample]).rxn_scores
            _, tasks_scores = self.rxn_tester.run(expression_threshold=expr_tol_dict[sample],
                                                  non_expression_threshold=nexpr_tol_dict[sample],
                                                  rxn_scores=rxn_scores,
                                                  ref_model=model_dict[sample],
                                                  reset_tester=True,
                                                  test_sink=test_sink)
            self.output[sample] = tasks_scores

        return self.output