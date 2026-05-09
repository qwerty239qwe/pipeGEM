"""Input validation utilities for API boundaries."""
from typing import List, Optional

import cobra
import pandas as pd

from pipeGEM.exceptions import (
    ModelValidationError,
    DataValidationError,
    DataAlignmentError,
)


def validate_cobra_model(model, param_name: str = "model") -> cobra.Model:
    """Validate that *model* is a ``cobra.Model`` and return it.

    If a ``pipeGEM.Model`` is passed, the underlying ``cobra.Model`` is
    extracted and returned.

    Parameters
    ----------
    model : object
        The object to validate.
    param_name : str
        Name of the parameter (used in error messages).

    Returns
    -------
    cobra.Model

    Raises
    ------
    ModelValidationError
        If *model* is not a ``cobra.Model`` (or ``pipeGEM.Model``).
    """
    # Avoid circular import – pipeGEM.Model wraps cobra.Model
    if hasattr(model, "cobra_model"):
        model = model.cobra_model
    if not isinstance(model, cobra.Model):
        raise ModelValidationError(
            f"'{param_name}' must be a cobra.Model or pipeGEM.Model, "
            f"got {type(model).__name__}."
        )
    return model


def validate_dataframe(
    data,
    required_columns: Optional[List[str]] = None,
    param_name: str = "data",
) -> pd.DataFrame:
    """Validate that *data* is a :class:`~pandas.DataFrame`.

    Parameters
    ----------
    data : object
        The object to validate.
    required_columns : list of str, optional
        Column names that must be present.
    param_name : str
        Name of the parameter (used in error messages).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    DataValidationError
        If *data* is not a DataFrame or is missing required columns.
    """
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError(
            f"'{param_name}' must be a pandas DataFrame, "
            f"got {type(data).__name__}."
        )
    if required_columns:
        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            raise DataValidationError(
                f"'{param_name}' is missing required column(s): {missing}."
            )
    return data


def validate_gene_ids(
    gene_ids: List[str],
    model: cobra.Model,
    param_name: str = "gene_ids",
    min_overlap: float = 0.0,
) -> List[str]:
    """Validate that *gene_ids* overlap with genes in *model*.

    Parameters
    ----------
    gene_ids : list of str
        Gene IDs to validate.
    model : cobra.Model
        Reference model.
    param_name : str
        Name of the parameter (used in error messages).
    min_overlap : float
        Minimum fraction of *gene_ids* that must appear in *model*
        (0.0 – 1.0).  If the actual overlap is lower, a
        :class:`~pipeGEM.exceptions.DataAlignmentError` is raised.

    Returns
    -------
    list of str
        The subset of *gene_ids* present in *model*.

    Raises
    ------
    DataAlignmentError
        If overlap is below *min_overlap*.
    """
    model_gene_set = {g.id for g in model.genes}
    overlap = [gid for gid in gene_ids if gid in model_gene_set]
    if len(gene_ids) > 0 and len(overlap) / len(gene_ids) < min_overlap:
        raise DataAlignmentError(
            f"Only {len(overlap)}/{len(gene_ids)} IDs in '{param_name}' "
            f"match genes in the model (minimum overlap: {min_overlap:.0%})."
        )
    return overlap


def validate_reaction_ids(
    rxn_ids: List[str],
    model: cobra.Model,
    param_name: str = "rxn_ids",
    min_overlap: float = 0.0,
) -> List[str]:
    """Validate that *rxn_ids* overlap with reactions in *model*.

    Parameters
    ----------
    rxn_ids : list of str
        Reaction IDs to validate.
    model : cobra.Model
        Reference model.
    param_name : str
        Name of the parameter (used in error messages).
    min_overlap : float
        Minimum fraction of *rxn_ids* that must appear in *model*
        (0.0 – 1.0).

    Returns
    -------
    list of str
        The subset of *rxn_ids* present in *model*.

    Raises
    ------
    DataAlignmentError
        If overlap is below *min_overlap*.
    """
    model_rxn_set = {r.id for r in model.reactions}
    overlap = [rid for rid in rxn_ids if rid in model_rxn_set]
    if len(rxn_ids) > 0 and len(overlap) / len(rxn_ids) < min_overlap:
        raise DataAlignmentError(
            f"Only {len(overlap)}/{len(rxn_ids)} IDs in '{param_name}' "
            f"match reactions in the model (minimum overlap: {min_overlap:.0%})."
        )
    return overlap
