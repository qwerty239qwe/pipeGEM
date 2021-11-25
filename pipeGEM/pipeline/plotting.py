from typing import List

import pandas as pd

from pipeGEM.pipeline import Pipeline
from .preprocessing import GeneDataSet
from pipeGEM.plotting.heatmap import plot_clustermap


class TestScorePlotter(Pipeline):
    def __init__(self, model_tester):
        super().__init__()
        self.model_tester = model_tester

    def run(self,
            task_scores: List[str],
            file_name=None,
            z_score=1) -> None:
        score_data = pd.DataFrame(task_scores).fillna(-1)
        subsystem_dict = {ID: task.subsystem
                          for ID, task in self.model_tester.tasks.items()}
        plot_clustermap(score_data,
                        row_category=subsystem_dict,
                        file_name=file_name,
                        z_score=z_score,
                        )


class GeneDataSetHeatmap(Pipeline):
    def __init__(self, data_df, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ds = GeneDataSet(data_df=data_df)

    def run(self,
            model,
            level="subsystems",
            statistics="mean",
            plotting_kws=None,
            *args,
            **kwargs) -> None:
        exp_dic = self.ds(model, *args, **kwargs)
        dfs = []
        plotting_kws = plotting_kws if plotting_kws is not None else {}
        for sample_name, exp in exp_dic.items():
            if level == "subsystems":
                scores = exp.get_subsystem_scores(statistics=statistics)
            elif level == "rxns":
                scores = exp.rxn_scores
            else:
                raise ValueError("choose from subsystems or rxns")
            dfs.append(pd.Series(scores).to_frame().rename(columns={0: sample_name}))
        plot_clustermap(pd.concat(dfs, axis=1), **plotting_kws)


