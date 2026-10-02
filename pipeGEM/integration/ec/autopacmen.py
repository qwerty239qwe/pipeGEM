"""AutoPACMEN-style automated parameter collection pipeline.

Fetches kcat values from databases, matches them to model reactions via
EC numbers, fills missing values, and returns an enriched
:class:`~pipeGEM.data.EnzymeData` ready for GECKO.
"""
import warnings
from typing import Optional, Literal

import numpy as np
import pandas as pd

from pipeGEM._logging import get_logger
from pipeGEM.data.data import EnzymeData

logger = get_logger(__name__)


def _positive_kcats(values):
    values = pd.to_numeric(values, errors="coerce")
    return values.where(np.isfinite(values) & (values > 0))


def _fill_kcats(values, strategy="median"):
    values = _positive_kcats(values)
    valid = values.dropna()
    if valid.empty:
        fill_value = 1.0
        logger.warning("No finite positive kcats available; using placeholder 1.0 s^-1.")
    else:
        fill_value = np.exp(np.log(valid).mean()) if strategy == "geometric_mean" else valid.median()
    return values.fillna(fill_value)


def auto_parameterize(
    model,
    enzyme_data: EnzymeData,
    kcat_source: Literal["brenda", "sabio-rk", "manual"] = "manual",
    fill_missing: Literal["median", "geometric_mean", "dlkcat"] = "median",
    organism: str = "human",
    metabolite_data=None,
    device: str = "cpu",
) -> EnzymeData:
    """Automated parameter collection and estimation pipeline.

    Steps
    -----
    1. (Optional) Fetch kcat values from a database (BRENDA, SABIO-RK).
    2. Match to model reactions via EC numbers.
    3. Fill missing kcat values using the specified strategy.
    4. Optionally use DLKcat for prediction of remaining missing values.
    5. Return the enriched :class:`EnzymeData`.

    Parameters
    ----------
    model : cobra.Model
        The metabolic model.
    enzyme_data : EnzymeData
        Existing enzyme data (may have missing kcat values).
    kcat_source : str
        Source for kcat values: ``"brenda"``, ``"sabio-rk"``, or
        ``"manual"`` (use only what is already in *enzyme_data*).
    fill_missing : str
        Strategy to fill missing kcat values:
        ``"median"`` — use the median of available kcats,
        ``"geometric_mean"`` — use the geometric mean,
        ``"dlkcat"`` — use DLKcat deep-learning prediction.
    organism : str
        Organism name (used for database queries).
    metabolite_data : pipeGEM.data.MetaboliteData, optional
        Metabolite data with SMILES (required when *fill_missing* is
        ``"dlkcat"``).
    device : str
        Device for DLKcat (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    EnzymeData
        The enriched enzyme data with finite positive kcat values. Zero,
        negative and infinite kcats are treated as missing. Fill statistics
        use finite positive values, or 1.0 s^-1 if none are available.
    """
    df = enzyme_data._enzyme_df.copy()
    kcat_col = enzyme_data.kcat_col
    df[kcat_col] = _positive_kcats(df[kcat_col])

    # ------------------------------------------------------------------
    # Step 1: Fetch from database (if requested)
    # ------------------------------------------------------------------
    if kcat_source == "brenda":
        logger.info("Fetching kcat values from BRENDA for organism '%s'...", organism)
        try:
            from pipeGEM.data.fetching import fetch_brenda_kcat
            brenda_df = fetch_brenda_kcat(organism=organism)
            df = _merge_kcat_from_source(df, brenda_df, kcat_col, enzyme_data.ec_num_col)
        except (ImportError, Exception) as exc:
            logger.warning("BRENDA fetch failed: %s. Continuing with existing data.", exc)

    elif kcat_source == "sabio-rk":
        logger.info("Fetching kcat values from SABIO-RK for organism '%s'...", organism)
        warnings.warn("SABIO-RK fetching is not yet fully implemented. Using existing data.")

    # ------------------------------------------------------------------
    # Step 2: Fill missing values
    # ------------------------------------------------------------------
    n_missing_before = df[kcat_col].isna().sum()
    logger.info("Missing kcat values before filling: %d / %d", n_missing_before, len(df))

    if fill_missing in ("median", "geometric_mean"):
        df[kcat_col] = _fill_kcats(df[kcat_col], fill_missing)

    elif fill_missing == "dlkcat":
        if metabolite_data is None:
            warnings.warn(
                "DLKcat requested but metabolite_data is None. "
                "Falling back to median filling."
            )
        else:
            logger.info("Running DLKcat prediction for missing kcats...")
            try:
                enzyme_data.run_DLKcat(metabolite_data, device=device)
                # After DLKcat, use alt_kcat_col to fill gaps
                alt_col = enzyme_data.alt_kcat_col
                if alt_col in enzyme_data._enzyme_df.columns:
                    df[alt_col] = enzyme_data._enzyme_df[alt_col]
                    df[kcat_col] = df[kcat_col].fillna(_positive_kcats(df[alt_col]))
            except Exception as exc:
                logger.warning("DLKcat prediction failed: %s. Using median.", exc)
        df[kcat_col] = _fill_kcats(df[kcat_col])
    else:
        raise ValueError(f"Unknown kcat fill strategy: {fill_missing}")

    n_missing_after = df[kcat_col].isna().sum()
    logger.info(
        "Missing kcat values after filling: %d / %d (was %d)",
        n_missing_after, len(df), n_missing_before,
    )

    # ------------------------------------------------------------------
    # Step 3: Rebuild EnzymeData with enriched DataFrame
    # ------------------------------------------------------------------
    enriched = EnzymeData(
        data=df,
        gene_id_col=None,  # index is already gene IDs
        prot_id_col=enzyme_data.prot_id_col,
        rxn_id_col=enzyme_data._rxn_id_col,
        met_id_col=enzyme_data._met_id_col,
        mw_col=enzyme_data.mw_col,
        kcat_col=enzyme_data.kcat_col,
        alt_kcat_col=enzyme_data.alt_kcat_col,
        prot_seq_col=enzyme_data.prot_seq_col,
        ec_num_col=enzyme_data.ec_num_col,
        sa_col=enzyme_data.sa_col,
    )
    return enriched


def _merge_kcat_from_source(df, source_df, kcat_col, ec_col):
    """Merge kcat values from a source DataFrame using EC number matching."""
    if ec_col not in df.columns or "EC" not in source_df.columns:
        logger.warning("Cannot merge: EC column missing.")
        return df

    # Only fill where kcat is missing
    df[kcat_col] = _positive_kcats(df[kcat_col])
    missing_mask = df[kcat_col].isna()
    if not missing_mask.any():
        return df

    source_df = source_df.assign(kcat=_positive_kcats(source_df["kcat"]))
    ec_to_kcat = source_df.groupby("EC")["kcat"].median().to_dict()
    for idx in df[missing_mask].index:
        ec = df.loc[idx, ec_col]
        if ec in ec_to_kcat:
            df.loc[idx, kcat_col] = ec_to_kcat[ec]

    n_filled = missing_mask.sum() - df[kcat_col].isna().sum()
    logger.info("Filled %d kcat values from external source.", n_filled)
    return df
