from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

from .base import ConfigSource, RawConfig


@dataclass(frozen=True)
class GeneDataConfig(RawConfig):
    REQUIRED_KEYS: ClassVar[tuple[str, ...]] = ("input.input_file_path", "params")

    @property
    def input_file_path(self) -> Path:
        return Path(self.require("input.input_file_path"))

    @property
    def input_kwargs(self) -> dict[str, Any]:
        data = dict(self.require("input"))
        data.pop("input_file_path", None)
        return data

    @property
    def params(self) -> Mapping[str, Any]:
        return self.require("params")


@dataclass(frozen=True)
class ModelProcessingConfig(RawConfig):
    REQUIRED_KEYS: ClassVar[tuple[str, ...]] = (
        "input.input_file_path",
        "param",
        "medium_data.name",
        "medium_data.apply_before",
        "medium_data.apply_after",
        "medium_data.params",
        "medium_data.apply_params",
        "rescale.method",
        "rescale.n_iter",
        "rescale.saved_path",
        "consistency.method",
        "consistency.saved_path",
        "consistency.params",
        "functionality_test.saved_path",
        "functionality_test.params.tasks_file_name",
        "functionality_test.test_tasks",
    )

    @property
    def input_file_path(self) -> Path:
        return Path(self.require("input.input_file_path"))


@dataclass(frozen=True)
class ThresholdConfig(RawConfig):
    REQUIRED_KEYS: ClassVar[tuple[str, ...]] = ("saved_path", "params.name")

    @property
    def name(self) -> str:
        return self.require("params.name")


@dataclass(frozen=True)
class MappingConfig(RawConfig):
    REQUIRED_KEYS: ClassVar[tuple[str, ...]] = (
        "threshold_analysis.type",
        "threshold_analysis.input_file_path_pattern",
        "rxn_score.align",
        "task_score.input_file_path",
        "task_score.get_supp_rxns",
    )


@dataclass(frozen=True)
class IntegrationConfig(RawConfig):
    REQUIRED_KEYS: ClassVar[tuple[str, ...]] = (
        "integrator_name",
        "saved_path",
        "precompute.model.cobra_model_path",
        "precompute.model.model_name_tag",
        "precompute.tasks.task_result_path",
        "precompute.threshold.threshold_result_path",
        "precompute.threshold.type",
    )

    @property
    def integrator_name(self) -> str:
        return self.require("integrator_name")


@dataclass(frozen=True)
class FluxAnalysisConfig(RawConfig):
    REQUIRED_KEYS: ClassVar[tuple[str, ...]] = ("saved_path",)


@dataclass(frozen=True)
class MultiModelConfig(RawConfig):
    REQUIRED_KEYS: ClassVar[tuple[str, ...]] = ("model_type",)


@dataclass(frozen=True)
class ComparisonConfig(RawConfig):
    REQUIRED_KEYS: ClassVar[tuple[str, ...]] = (
        "models_input_path",
        "model_type",
        "factor_file",
        "output_dir",
        "compare_num.output_file_name",
        "compare_num.group",
        "compare_num.dpi",
        "compare_jaccard.output_file_name",
        "compare_jaccard.row_color_by",
        "compare_jaccard.col_color_by",
        "compare_jaccard.dpi",
        "compare_PCA.output_file_name",
        "compare_PCA.color_by",
        "compare_PCA.dpi",
    )


def from_mapping(raw: Mapping[str, Any], label: str, config_cls: type[RawConfig]) -> RawConfig:
    return config_cls(raw=raw, source=ConfigSource(label=label))
