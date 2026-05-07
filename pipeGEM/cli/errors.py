class CLIError(Exception):
    """Base class for user-facing CLI errors."""


class ConfigError(CLIError, ValueError):
    """Raised when a CLI configuration file is missing required data."""


class PipelineError(CLIError, RuntimeError):
    """Raised when a CLI pipeline cannot be planned or executed."""
