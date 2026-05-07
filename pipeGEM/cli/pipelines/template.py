from __future__ import annotations

from pathlib import Path

from pipeGEM.cli.config import TemplatePipelineConfig
from pipeGEM.cli.errors import PipelineError
from pipeGEM.cli.pipelines.base import BasePipeline, PipelinePlan, PipelineResult, PipelineStep


class TemplatePipeline(BasePipeline):
    name = "template"

    def __init__(self, config: TemplatePipelineConfig):
        self.config = config

    def validate(self) -> None:
        from pipeGEM.cli._utils import pl_needed_config

        if self.config.pipeline not in pl_needed_config:
            valid = ", ".join(sorted(pl_needed_config))
            raise PipelineError(f"unknown template pipeline '{self.config.pipeline}'. Valid pipelines: {valid}")

    def plan(self) -> PipelinePlan:
        self.validate()
        return PipelinePlan(
            name=self.name,
            steps=(PipelineStep(
                name=f"generate {self.config.pipeline} template configs",
                outputs=(Path(self.config.output_path) / "configs",),
            ),),
        )

    def run(self) -> PipelineResult:
        from pipeGEM.cli._utils import generate_template_configs

        self.validate()
        generate_template_configs(dest_folder=str(self.config.output_path), pl_name=self.config.pipeline)
        return PipelineResult(name=self.name, outputs=(Path(self.config.output_path) / "configs",))
