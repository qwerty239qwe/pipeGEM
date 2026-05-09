from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineStep:
    name: str
    inputs: tuple[Path, ...] = field(default_factory=tuple)
    outputs: tuple[Path, ...] = field(default_factory=tuple)
    skipped: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class PipelinePlan:
    name: str
    steps: tuple[PipelineStep, ...]

    def format(self) -> str:
        lines = [f"Pipeline: {self.name}"]
        for step in self.steps:
            suffix = f" (skip: {step.reason})" if step.skipped and step.reason else ""
            lines.append(f"- {step.name}{suffix}")
            if step.inputs:
                lines.append(f"  inputs: {', '.join(str(p) for p in step.inputs)}")
            if step.outputs:
                lines.append(f"  outputs: {', '.join(str(p) for p in step.outputs)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class PipelineResult:
    name: str
    outputs: tuple[Path, ...] = field(default_factory=tuple)
    skipped: tuple[PipelineStep, ...] = field(default_factory=tuple)


class BasePipeline:
    name = "pipeline"

    def validate(self) -> None:
        return None

    def plan(self) -> PipelinePlan:
        raise NotImplementedError

    def run(self) -> PipelineResult:
        raise NotImplementedError
