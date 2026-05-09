"""Centralized logging configuration for pipeGEM."""
import logging
import sys
from typing import Optional

_PACKAGE_LOGGER_NAME = "pipeGEM"
_DEFAULT_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Get a logger within the pipeGEM namespace.

    Parameters
    ----------
    name : str
        The name of the logger, typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(f"{_PACKAGE_LOGGER_NAME}.{name}")


def set_log_level(level: int = logging.WARNING) -> None:
    """Set the log level for all pipeGEM loggers.

    Parameters
    ----------
    level : int
        A logging level, e.g. ``logging.DEBUG``, ``logging.INFO``,
        ``logging.WARNING`` (default).
    """
    logging.getLogger(_PACKAGE_LOGGER_NAME).setLevel(level)


def enable_verbose(handler: Optional[logging.Handler] = None) -> None:
    """Enable verbose (DEBUG) output for pipeGEM.

    Parameters
    ----------
    handler : logging.Handler, optional
        A custom handler. If ``None``, a ``StreamHandler`` writing to
        ``sys.stderr`` is created with the default format.
    """
    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    if not logger.handlers:
        logger.addHandler(handler)


# Initialize the package-level logger with a NullHandler so that library
# users are not forced to configure logging.  The default level is WARNING.
_root_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
_root_logger.addHandler(logging.NullHandler())
_root_logger.setLevel(logging.WARNING)
