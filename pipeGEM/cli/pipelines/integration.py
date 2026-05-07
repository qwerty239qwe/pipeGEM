from __future__ import annotations

from pathlib import Path

from pipeGEM.cli.config import IntegrationPipelineConfig
from pipeGEM.cli.pipelines.base import BasePipeline, PipelinePlan, PipelineResult, PipelineStep


class IntegrationPipeline(BasePipeline):
    name = "integration"

    def __init__(self, config: IntegrationPipelineConfig):
        self.config = config

    def plan(self) -> PipelinePlan:
        return PipelinePlan(
            name=self.name,
            steps=(
                PipelineStep(
                    name="load gene data",
                    inputs=(self.config.gene_data.input_file_path,),
                ),
                PipelineStep(
                    name="prepare model and task support reactions",
                    inputs=(self.config.model.input_file_path,),
                ),
                PipelineStep(
                    name=f"integrate gene data with {self.config.integration.integrator_name}",
                    outputs=(Path(self.config.integration.require("saved_path")),),
                ),
            ),
        )

    def run(self) -> PipelineResult:
        from pipeGEM.cli._utils import run_integration_pipeline

        self.validate()
        output = Path(self.config.integration.require("saved_path"))
        run_integration_pipeline(
            gene_data_conf=self.config.gene_data.as_dict(),
            model_conf=self.config.model.as_dict(),
            threshold_conf=self.config.threshold.as_dict(),
            mapping_conf=self.config.mapping.as_dict(),
            integration_conf=self.config.integration.as_dict(),
        )
        return PipelineResult(name=self.name, outputs=(output,))
