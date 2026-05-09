from __future__ import annotations

from pathlib import Path

from pipeGEM.cli.config import ThresholdPipelineConfig
from pipeGEM.cli.pipelines.base import BasePipeline, PipelinePlan, PipelineResult, PipelineStep


class ThresholdPipeline(BasePipeline):
    name = "threshold"

    def __init__(self, config: ThresholdPipelineConfig):
        self.config = config

    def plan(self) -> PipelinePlan:
        return PipelinePlan(
            name=self.name,
            steps=(PipelineStep(
                name=f"compute {self.config.threshold.name} thresholds",
                inputs=(self.config.gene_data.input_file_path,),
                outputs=(Path(self.config.threshold.require("saved_path")),),
            ),),
        )

    def run(self) -> PipelineResult:
        from pipeGEM.cli._io import load_gene_data
        from pipeGEM.cli._utils import find_threshold

        self.validate()
        gene_data_dic = load_gene_data(gene_data_conf=self.config.gene_data.as_dict())
        find_threshold(gene_data_dic, self.config.threshold.as_dict())
        return PipelineResult(name=self.name, outputs=(Path(self.config.threshold.require("saved_path")),))
