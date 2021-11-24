from typing import List, Dict, Optional

import pandas as pd
import cobra

from ._base import Pipeline
from pipeGEM.analysis.tasks import TaskTester, TASKS_FILE_PATH


class ReactionTester(Pipeline):
    def __init__(self,
                 task_tester: Optional[TaskTester] = None,
                 ref_model: Optional[cobra.Model] = None,
                 task_file_path=TASKS_FILE_PATH,
                 model_compartment_format="[{}]",
                 constr_name = "default",
                 **kwargs):
        super().__init__()
        self.constr_name: str = constr_name
        self.model_tester = TaskTester(ref_model,
                                       constr=constr_name,
                                       task_container=task_file_path,
                                       model_compartment_parenthesis=model_compartment_format,
                                       **kwargs) if task_tester is not None else task_tester

    def run(self,
            expression_threshold: float,
            non_expression_threshold: float,
            rxn_scores: dict) -> (List[str], Dict[str, float]):
        self.model_tester.update_thresholds(express_thres=expression_threshold,
                                            non_express_thres=non_expression_threshold,
                                            rxn_scores=rxn_scores)
        self.model_tester.test()
        self.model_tester.map_expr_to_tasks()
        return self.model_tester.get_all_passed_rxns(), self.model_tester.tasks_scores

# add validator


class TaskScoringPipeLine(Pipeline):
    def __init__(self):
        super().__init__()

    def run(self, *args, **kwargs):
        pass