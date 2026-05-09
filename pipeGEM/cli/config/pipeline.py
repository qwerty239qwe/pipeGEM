from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import load_config_file
from .types import (
    ComparisonConfig,
    FluxAnalysisConfig,
    GeneDataConfig,
    IntegrationConfig,
    MappingConfig,
    ModelProcessingConfig,
    MultiModelConfig,
    ThresholdConfig,
)


@dataclass(frozen=True)
class TemplatePipelineConfig:
    pipeline: str
    output_path: Path


@dataclass(frozen=True)
class ModelProcessingPipelineConfig:
    model: ModelProcessingConfig

    @classmethod
    def from_files(cls, model: str | Path) -> "ModelProcessingPipelineConfig":
        return cls(model=load_config_file(model, "model", ModelProcessingConfig))


@dataclass(frozen=True)
class ThresholdPipelineConfig:
    gene_data: GeneDataConfig
    threshold: ThresholdConfig

    @classmethod
    def from_files(cls, gene_data: str | Path, threshold: str | Path) -> "ThresholdPipelineConfig":
        return cls(
            gene_data=load_config_file(gene_data, "gene_data", GeneDataConfig),
            threshold=load_config_file(threshold, "threshold", ThresholdConfig),
        )


@dataclass(frozen=True)
class IntegrationPipelineConfig:
    gene_data: GeneDataConfig
    model: ModelProcessingConfig
    threshold: ThresholdConfig
    mapping: MappingConfig
    integration: IntegrationConfig

    @classmethod
    def from_files(
        cls,
        gene_data: str | Path,
        model: str | Path,
        threshold: str | Path,
        mapping: str | Path,
        integration: str | Path,
    ) -> "IntegrationPipelineConfig":
        return cls(
            gene_data=load_config_file(gene_data, "gene_data", GeneDataConfig),
            model=load_config_file(model, "model", ModelProcessingConfig),
            threshold=load_config_file(threshold, "threshold", ThresholdConfig),
            mapping=load_config_file(mapping, "mapping", MappingConfig),
            integration=load_config_file(integration, "integration", IntegrationConfig),
        )


@dataclass(frozen=True)
class FluxPipelineConfig:
    flux_analysis: FluxAnalysisConfig
    model: MultiModelConfig
    gene_data: GeneDataConfig | None = None
    threshold: ThresholdConfig | None = None
    mapping: MappingConfig | None = None
    integration: IntegrationConfig | None = None

    @classmethod
    def from_files(
        cls,
        flux_analysis: str | Path,
        model: str | Path,
        gene_data: str | Path | None = None,
        threshold: str | Path | None = None,
        mapping: str | Path | None = None,
        integration: str | Path | None = None,
    ) -> "FluxPipelineConfig":
        return cls(
            flux_analysis=load_config_file(flux_analysis, "flux_analysis", FluxAnalysisConfig),
            model=load_config_file(model, "model", MultiModelConfig),
            gene_data=load_config_file(gene_data, "gene_data", GeneDataConfig) if gene_data else None,
            threshold=load_config_file(threshold, "threshold", ThresholdConfig) if threshold else None,
            mapping=load_config_file(mapping, "mapping", MappingConfig) if mapping else None,
            integration=load_config_file(integration, "integration", IntegrationConfig) if integration else None,
        )


@dataclass(frozen=True)
class ComparisonPipelineConfig:
    comparison: ComparisonConfig

    @classmethod
    def from_files(cls, comparison: str | Path) -> "ComparisonPipelineConfig":
        return cls(comparison=load_config_file(comparison, "comparison", ComparisonConfig))
