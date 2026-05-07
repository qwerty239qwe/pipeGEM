from __future__ import annotations

from pathlib import Path

from pipeGEM.cli.config import FluxPipelineConfig
from pipeGEM.cli.errors import PipelineError
from pipeGEM.cli.pipelines.base import BasePipeline, PipelinePlan, PipelineResult, PipelineStep


class FluxPipeline(BasePipeline):
    name = "flux"

    def __init__(self, config: FluxPipelineConfig):
        self.config = config

    def validate(self) -> None:
        integration_related = {
            "gene_data": self.config.gene_data,
            "threshold": self.config.threshold,
            "mapping": self.config.mapping,
            "integration": self.config.integration,
        }
        supplied = {name for name, value in integration_related.items() if value is not None}
        if supplied and supplied != set(integration_related):
            missing = ", ".join(sorted(set(integration_related) - supplied))
            raise PipelineError(f"flux analysis with integration requires all integration configs; missing: {missing}")
        if self.config.integration is None:
            self.config.model.require("models_input_dir")
            self.config.model.require("group_factor_path")
        else:
            self.config.model.require("models_input_path")

    def plan(self) -> PipelinePlan:
        self.validate()
        inputs = []
        for item in (self.config.gene_data, self.config.threshold, self.config.mapping, self.config.integration):
            if item is not None and item.source.path is not None:
                inputs.append(item.source.path)
        if self.config.model.source.path is not None:
            inputs.append(self.config.model.source.path)
        return PipelinePlan(
            name=self.name,
            steps=(PipelineStep(
                name="run flux analysis",
                inputs=tuple(inputs),
                outputs=(Path(self.config.flux_analysis.require("saved_path")),),
            ),),
        )

    def run(self) -> PipelineResult:
        from pipeGEM.cli._utils import do_flux_analysis

        self.validate()
        output = Path(self.config.flux_analysis.require("saved_path"))
        do_flux_analysis(
            fa_configs=self.config.flux_analysis.as_dict(),
            multi_model_conf=self.config.model.as_dict(),
            gene_data_conf=self.config.gene_data.as_dict() if self.config.gene_data else None,
            threshold_conf=self.config.threshold.as_dict() if self.config.threshold else None,
            mapping_conf=self.config.mapping.as_dict() if self.config.mapping else None,
            integration_conf=self.config.integration.as_dict() if self.config.integration else None,
        )
        return PipelineResult(name=self.name, outputs=(output,))
