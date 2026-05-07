from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

from pipeGEM.cli.errors import ConfigError
from pipeGEM.utils import parse_toml_file


@dataclass(frozen=True)
class ConfigSource:
    """Where a config came from, used for clearer error messages."""

    label: str
    path: Path | None = None

    def describe(self) -> str:
        if self.path is None:
            return self.label
        return f"{self.label} config ({self.path})"


@dataclass(frozen=True)
class RawConfig:
    """Validated wrapper around a TOML-compatible config dictionary."""

    REQUIRED_KEYS: ClassVar[tuple[str, ...]] = ()

    raw: Mapping[str, Any]
    source: ConfigSource

    def __post_init__(self) -> None:
        for key_path in self.REQUIRED_KEYS:
            self.require(key_path)

    def as_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self.raw))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], label: str) -> "RawConfig":
        return cls(raw=raw, source=ConfigSource(label=label))

    @classmethod
    def from_toml(cls, path: str | Path, label: str) -> "RawConfig":
        return load_config_file(path, label, cls)

    def get(self, key_path: str, default: Any = None) -> Any:
        found, value = self._lookup(key_path)
        return value if found else default

    def _lookup(self, key_path: str) -> tuple[bool, Any]:
        current: Any = self.raw
        for key in key_path.split("."):
            if not isinstance(current, Mapping) or key not in current:
                return False, None
            current = current[key]
        return True, current

    def require(self, key_path: str) -> Any:
        found, value = self._lookup(key_path)
        if not found:
            raise ConfigError(f"missing config key '{key_path}' in {self.source.describe()}")
        return value


def load_config_file(path: str | Path, label: str, config_cls: type[RawConfig] = RawConfig) -> RawConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"{label} config file not found: {path}")
    return config_cls(raw=parse_toml_file(path), source=ConfigSource(label=label, path=path))
