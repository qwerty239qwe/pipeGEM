"""Pathway-level flux aggregation, comparison, and bottleneck analysis."""
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from pipeGEM._logging import get_logger

logger = get_logger(__name__)


class PathwayAnalyzer:
    """Pathway-level flux aggregation and comparison.

    Parameters
    ----------
    model : cobra.Model
        The metabolic model.
    pathway_definitions : dict
        Mapping of pathway names/IDs to lists of reaction IDs.
    """

    def __init__(self, model, pathway_definitions: Dict[str, List[str]]):
        self.model = model
        self.pathways = pathway_definitions

    def aggregate_fluxes(
        self,
        flux_df: pd.DataFrame,
        method: Literal["mean", "median", "sum", "max", "min"] = "mean",
    ) -> pd.DataFrame:
        """Aggregate reaction fluxes to pathway level.

        Parameters
        ----------
        flux_df : pandas.DataFrame
            Flux values indexed by reaction ID. Can have multiple columns
            (e.g. different samples/conditions).
        method : str
            Aggregation method.

        Returns
        -------
        pandas.DataFrame
            Pathway-level aggregated fluxes.
        """
        rows = []
        for pathway_id, rxn_ids in self.pathways.items():
            available = [r for r in rxn_ids if r in flux_df.index]
            if not available:
                continue
            subset = flux_df.loc[available]
            agg = getattr(subset, method)(axis=0)
            row = agg.to_dict() if isinstance(agg, pd.Series) else {"flux": agg}
            row["pathway"] = pathway_id
            row["n_reactions"] = len(available)
            rows.append(row)

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.set_index("pathway")
        return result

    def compare_pathways(
        self,
        flux_dfs: Dict[str, pd.DataFrame],
        method: Literal["mean", "median"] = "mean",
    ) -> pd.DataFrame:
        """Compare pathway-level fluxes between conditions.

        Parameters
        ----------
        flux_dfs : dict
            Mapping of condition names to flux DataFrames.
        method : str
            Aggregation method applied per condition.

        Returns
        -------
        pandas.DataFrame
            DataFrame with pathway rows and condition columns.
        """
        agg_frames = {}
        for cond_name, fdf in flux_dfs.items():
            agg = self.aggregate_fluxes(fdf, method=method)
            if not agg.empty:
                # Take first numeric column for comparison
                numeric_cols = agg.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    agg_frames[cond_name] = agg[numeric_cols[0]]
                elif "flux" in agg.columns:
                    agg_frames[cond_name] = agg["flux"]

        if not agg_frames:
            return pd.DataFrame()

        return pd.DataFrame(agg_frames)

    def identify_bottlenecks(
        self,
        fva_df: pd.DataFrame,
        min_col: str = "minimum",
        max_col: str = "maximum",
        threshold: float = 0.1,
    ) -> pd.DataFrame:
        """Identify pathway bottlenecks from FVA ranges.

        A bottleneck is a reaction with a narrow FVA range (max - min)
        relative to the pathway average, suggesting it limits pathway
        throughput.

        Parameters
        ----------
        fva_df : pandas.DataFrame
            FVA result with *min_col* and *max_col* columns.
        min_col, max_col : str
            Column names for FVA minimum and maximum.
        threshold : float
            Reactions whose range is below *threshold* times the pathway
            average range are flagged as bottlenecks.

        Returns
        -------
        pandas.DataFrame
            Bottleneck reactions with columns ``pathway``, ``reaction``,
            ``range``, ``pathway_avg_range``.
        """
        if min_col not in fva_df.columns or max_col not in fva_df.columns:
            raise ValueError(
                f"FVA DataFrame must have '{min_col}' and '{max_col}' columns."
            )

        fva_df = fva_df.copy()
        fva_df["_range"] = fva_df[max_col] - fva_df[min_col]

        rows = []
        for pathway_id, rxn_ids in self.pathways.items():
            available = [r for r in rxn_ids if r in fva_df.index]
            if not available:
                continue
            subset = fva_df.loc[available]
            avg_range = subset["_range"].mean()
            if avg_range == 0:
                continue
            for rxn_id in available:
                rxn_range = fva_df.loc[rxn_id, "_range"]
                if rxn_range < threshold * avg_range:
                    rows.append({
                        "pathway": pathway_id,
                        "reaction": rxn_id,
                        "range": rxn_range,
                        "pathway_avg_range": avg_range,
                        "ratio": rxn_range / avg_range if avg_range > 0 else 0,
                    })

        return pd.DataFrame(rows)
