"""GECKO-light: simple kcat-based enzyme constraints.

Applies ``kcat * enzyme_abundance`` as upper bounds on enzyme-catalysed
reactions, without modifying the model structure (no draw/arm reactions
or protein pool).
"""
import numpy as np
import pandas as pd

from pipeGEM._logging import get_logger
from pipeGEM.analysis.results._base import timing
from pipeGEM.analysis.results.ec import GECKOLightAnalysis
from pipeGEM.integration.ec._copy import copy_cobra_model

logger = get_logger(__name__)


def _check_gene_and_enzymes(model, enzyme_data):
    """Return gene IDs present in *model* but absent from *enzyme_data*."""
    gid_in_enzyme_data = {g.id: g.id in enzyme_data for g in model.genes}
    return [g for g, isin in gid_in_enzyme_data.items() if not isin]


@timing
def apply_gecko_light(
    model,
    enzyme_data,
    protein_abundance=None,
    sigma=0.5,
    f_factor=0.5,
    ptot=0.5,
    copy_model=True,
    protected_rxns=None,
):
    """Apply simple kcat-based enzyme constraints (GECKO-light).

    For every reaction in *model* that has associated kcat data, the absolute
    flux (both bounds, for reversible reactions) is constrained to::

        new_ub = kcat [1/s] * abundance [mmol/gDW] * sigma * 3600

    where the factor 3600 converts from per-second to per-hour to match
    typical COBRA flux units (mmol / gDW / h).
    Each reaction receives an independent cap, including reactions sharing
    an enzyme. Use full GECKO to share one enzyme budget across reactions.
    If absolute protein abundance is not provided, ``ptot * f_factor`` is
    used as a coarse fallback abundance scale so those parameters have a
    concrete effect.

    Parameters
    ----------
    model : cobra.Model
        The metabolic model to constrain.
    enzyme_data : pipeGEM.data.EnzymeData
        Enzyme data aligned with the model (must have been ``.align()``-ed).
    protein_abundance : pipeGEM.data.ProteinAbundanceData, optional
        Protein abundance data.  If ``None``, abundance is approximated as
        ``ptot * f_factor`` for every enzyme.
    sigma : float
        Average enzyme saturation factor (0 – 1).  Default 0.5.
    f_factor : float
        Fraction of the proteome that is metabolic enzymes (0 to 1). Used
        as part of the fallback abundance scale when protein abundance is not
        provided. Default 0.5.
    ptot : float
        Total protein content in g / gDW. Used as part of the fallback
        abundance scale when protein abundance is not provided. Default 0.5.
    copy_model : bool
        If ``True`` (default), work on a deep-copy of *model*.
    protected_rxns : list of str, optional
        Reaction IDs whose bounds should not be modified.

    Returns
    -------
    GECKOLightAnalysis

    Raises
    ------
    ValueError
        If enzyme capacity conflicts with a reaction's required flux.
        Bounds are validated before any changes are applied to the model.
    """
    if copy_model:
        model = copy_cobra_model(model)

    protected = set(protected_rxns or [])

    # Get kcat mapping from aligned enzyme data
    rxn_items = enzyme_data.rxn_items()

    modified_bounds = {}
    kcat_mapping = {}
    enzyme_usage_rows = []
    n_bound_reductions = 0

    for rxn_id, info in rxn_items.items():
        if rxn_id in protected:
            continue
        if rxn_id not in {r.id for r in model.reactions}:
            logger.debug("Reaction %s from enzyme data not in model, skipping.", rxn_id)
            continue

        rxn = model.reactions.get_by_id(rxn_id)
        kcat = info.get("best_kcat", None)
        if kcat is None or np.isnan(kcat) or kcat <= 0:
            logger.debug("No (or non-positive) kcat for reaction %s, skipping.", rxn_id)
            continue

        # Determine protein abundance. Absolute abundance data takes
        # precedence; otherwise use the global protein budget as a coarse
        # fallback so ptot and f_factor are not inert parameters.
        prot_id = info.get("protein_to_use", None)
        abundance = ptot * f_factor
        if protein_abundance is not None and prot_id is not None:
            prot_df = protein_abundance._prot_abund_df
            if prot_id in prot_df.index:
                measured = prot_df.loc[prot_id, protein_abundance.abundance_col]
                if pd.notna(measured):
                    abundance = measured

        # kcat is in 1/s -> convert to 1/h
        # new_ub = kcat [1/s] * abundance [mmol/gDW] * sigma * 3600 [s/h]
        new_ub = kcat * abundance * sigma * 3600.0

        # The enzyme caps |flux| in both directions, not only the forward one.
        old_lb, old_ub = rxn.bounds
        new_bounds = (max(old_lb, -new_ub), min(old_ub, new_ub))
        bound_reduced = new_bounds != (old_lb, old_ub)
        if new_bounds[0] > new_bounds[1]:
            raise ValueError(
                f"Enzyme capacity {new_ub:.4g} for {rxn_id} conflicts with bounds "
                f"({old_lb:.4g}, {old_ub:.4g}). Required flux exceeds enzyme capacity."
            )
        elif bound_reduced:
            n_bound_reductions += 1
            logger.debug(
                "Constrained %s: bounds (%.4g, %.4g) -> (%.4g, %.4g) (kcat=%.4g, abund=%.4g)",
                rxn_id, old_lb, old_ub, *new_bounds, kcat, abundance,
            )

        modified_bounds[rxn_id] = new_bounds
        kcat_mapping[rxn_id] = kcat
        enzyme_usage_rows.append({
            "reaction": rxn_id,
            "protein": prot_id,
            "kcat": kcat,
            "abundance": abundance,
            "new_ub": new_ub,
            "old_ub": old_ub,
            "bound_reduced": bound_reduced,
        })

    for rxn_id, bounds in modified_bounds.items():
        model.reactions.get_by_id(rxn_id).bounds = bounds

    enzyme_usage = pd.DataFrame(enzyme_usage_rows)
    logger.info(
        "GECKO-light applied: %d reactions with enzyme data, %d bounds reduced.",
        len(enzyme_usage), n_bound_reductions,
    )

    return GECKOLightAnalysis.from_results(
        log={
            "sigma": sigma,
            "f_factor": f_factor,
            "ptot": ptot,
            "n_reactions_with_enzyme_data": len(enzyme_usage),
            "n_bound_reductions": n_bound_reductions,
        },
        modified_bounds=modified_bounds,
        kcat_mapping=kcat_mapping,
        enzyme_usage=enzyme_usage,
        ec_model=model,
    )
