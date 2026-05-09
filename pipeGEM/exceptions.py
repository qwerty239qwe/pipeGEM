"""Custom exception hierarchy for pipeGEM.

All custom exceptions inherit from both :class:`PipeGEMError` and the
closest built-in exception type so that existing ``except ValueError``
(or similar) blocks continue to work.
"""


class PipeGEMError(Exception):
    """Base exception for all pipeGEM errors."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationError(PipeGEMError, ValueError):
    """Raised when input validation fails."""


class ModelValidationError(ValidationError):
    """Raised when a model object fails validation
    (e.g. not a ``cobra.Model``)."""


class DataValidationError(ValidationError):
    """Raised when supplied data fails validation
    (e.g. missing columns, wrong dtype)."""


# ---------------------------------------------------------------------------
# Data alignment
# ---------------------------------------------------------------------------

class DataAlignmentError(PipeGEMError, ValueError):
    """Raised when data cannot be aligned with a model
    (e.g. no overlapping gene IDs)."""


# ---------------------------------------------------------------------------
# Solver / LP
# ---------------------------------------------------------------------------

class SolverError(PipeGEMError, RuntimeError):
    """Raised when an LP / MILP solver encounters an error or
    returns an infeasible solution."""


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class IntegrationError(PipeGEMError, RuntimeError):
    """Raised when a data-integration algorithm fails."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ConfigurationError(PipeGEMError, ValueError):
    """Raised when a configuration file or parameter set is invalid."""


# ---------------------------------------------------------------------------
# Fitting / state
# ---------------------------------------------------------------------------

class NotFittedError(PipeGEMError, AttributeError):
    """Raised when a method is called before a required fitting or
    alignment step has been performed."""


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

class VisualizationError(PipeGEMError, RuntimeError):
    """Raised when a visualization cannot be produced
    (e.g. missing backend, invalid data for the requested plot)."""
