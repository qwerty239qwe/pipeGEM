from __future__ import annotations

from pathlib import Path

from pipeGEM.cli.config import ComparisonPipelineConfig
from pipeGEM.cli.pipelines.base import BasePipeline, PipelinePlan, PipelineResult, PipelineStep


class ComparisonPipeline(BasePipeline):
    name = "comparison"

    def __init__(self, config: ComparisonPipelineConfig):
        self.config = config

    def plan(self) -> PipelinePlan:
        return PipelinePlan(
            name=self.name,
            steps=(PipelineStep(
                name="compare metabolic models",
                inputs=(Path(self.config.comparison.require("models_input_path")),),
                outputs=(Path(self.config.comparison.require("output_dir")),),
            ),),
        )

    def run(self) -> PipelineResult:
        from pipeGEM.cli._utils import do_model_comparison

        self.validate()
        output = Path(self.config.comparison.require("output_dir"))
        do_model_comparison(comparison_configs=self.config.comparison.as_dict())
        return PipelineResult(name=self.name, outputs=(output,))
