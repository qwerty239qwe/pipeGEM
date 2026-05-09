from __future__ import annotations

from pathlib import Path

from pipeGEM.cli.config import ModelProcessingPipelineConfig
from pipeGEM.cli.pipelines.base import BasePipeline, PipelinePlan, PipelineResult, PipelineStep


class ModelProcessingPipeline(BasePipeline):
    name = "model_processing"

    def __init__(self, config: ModelProcessingPipelineConfig):
        self.config = config

    def plan(self) -> PipelinePlan:
        model_conf = self.config.model
        return PipelinePlan(
            name=self.name,
            steps=(PipelineStep(
                name="rescale, consistency check, and task-test model",
                inputs=(model_conf.input_file_path,),
                outputs=tuple(
                    Path(p) for p in (
                        model_conf.get("rescale.saved_path"),
                        model_conf.get("consistency.saved_path"),
                        model_conf.get("functionality_test.saved_path"),
                    )
                    if p
                ),
            ),),
        )

    def run(self) -> PipelineResult:
        from pipeGEM.cli._utils import preprocess_model

        self.validate()
        preprocess_model(model_conf=self.config.model.as_dict())
        outputs = tuple(output for step in self.plan().steps for output in step.outputs)
        return PipelineResult(name=self.name, outputs=outputs)
