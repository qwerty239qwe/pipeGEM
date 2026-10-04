"""Full GECKO: enzyme-constrained model with protein pool.

Creates a full ecModel by adding a protein pool pseudo-metabolite, draw
reactions from the pool to individual enzymes, and modifying metabolic
reactions to consume the enzyme pseudo-metabolite proportional to
``1 / kcat``.
"""
import numpy as np
import pandas as pd

from pipeGEM._logging import get_logger
from pipeGEM.analysis.results._base import timing
from pipeGEM.analysis.results.ec import GECKOFullAnalysis
from pipeGEM.integration.ec._builder import ECModelBuilder, PROT_POOL_ID
from pipeGEM.integration.ec._copy import copy_cobra_model

logger = get_logger(__name__)


@timing
def apply_gecko_full(
    model,
    enzyme_data,
    protein_abundance=None,
    sigma=0.5,
    ptot=0.5,
    f_factor=0.5,
    copy_model=True,
    protected_rxns=None,
):
    """Build a full enzyme-constrained model (ecModel).

    The GECKO formulation constrains total enzyme usage through a shared
    protein pool.  Each enzyme-catalysed reaction draws from this pool
    in proportion to ``MW / kcat``.

    Parameters
    ----------
    model : cobra.Model
        The metabolic model.
    enzyme_data : pipeGEM.data.EnzymeData
        Enzyme data aligned with the model.
    protein_abundance : pipeGEM.data.ProteinAbundanceData, optional
        Measured abundance in mmol/gDW. Each protein's draw reaction is
        capped at abundance * sigma, sharing capacity across its reactions.
        Missing IDs or NaN measurements retain the protein pool constraint.
    sigma : float
        Average enzyme saturation factor (0 – 1).
    ptot : float
        Total protein content in g / gDW.
    f_factor : float
        Fraction of the proteome that is metabolic enzymes (0 – 1).
    copy_model : bool
        Work on a deep-copy of *model* (default ``True``).
    protected_rxns : list of str, optional
        Reaction IDs whose bounds should not be modified.

    Returns
    -------
    GECKOFullAnalysis
    """
    if copy_model:
        model = copy_cobra_model(model)

    protected = set(protected_rxns or [])

    builder = ECModelBuilder(sigma=sigma, ptot=ptot, f_factor=f_factor)

    # 1. Add protein pool
    prot_pool = builder.add_protein_pool(model)

    # 2. For each reaction with kcat data, create draw + arm structures
    rxn_items = enzyme_data.rxn_items()

    n_enzyme_constraints = 0
    for rxn_id, info in rxn_items.items():
        if rxn_id in protected:
            continue
        if rxn_id not in {r.id for r in model.reactions}:
            logger.debug("Reaction %s not in model, skipping.", rxn_id)
            continue

        kcat = info.get("best_kcat", None)
        mw = info.get("best_mw", None)
        prot_id = info.get("protein_to_use", None)

        if kcat is None or np.isnan(kcat) or kcat <= 0:
            logger.debug("Invalid kcat for %s, skipping.", rxn_id)
            continue
        if mw is None or np.isnan(mw) or mw <= 0:
            logger.debug("Invalid MW for %s, skipping.", rxn_id)
            continue
        if prot_id is None or pd.isna(prot_id):
            # fallback; a NaN id would merge every such rxn into one "prot_nan" enzyme
            prot_id = rxn_id

        rxn = model.reactions.get_by_id(rxn_id)
        constrained_rxns = builder.prepare_reaction_for_enzyme_constraint(model, rxn)
        if not constrained_rxns:
            logger.debug("Reaction %s has no feasible flux direction, skipping.", rxn_id)
            continue

        # Create draw reaction (pool -> individual enzyme)
        # EnzymeData MW is in Da (g/mol); pool is g/gDW and enzyme usage is
        # mmol/gDW, so the draw coefficient must be g/mmol (kDa).
        enz_met = builder.create_draw_reaction(model, prot_pool, prot_id, mw / 1000.0)
        if protein_abundance is not None and prot_id in protein_abundance._prot_abund_df.index:
            abundance = protein_abundance._prot_abund_df.loc[prot_id, protein_abundance.abundance_col]
            if pd.notna(abundance):
                draw_rxn = model.reactions.get_by_id(f"draw_{prot_id}")
                draw_rxn.upper_bound = min(draw_rxn.upper_bound, abundance * sigma)

        # Modify each non-negative directional reaction to consume the enzyme
        for constrained_rxn in constrained_rxns:
            builder.create_arm_reaction(model, constrained_rxn, enz_met, kcat)
            n_enzyme_constraints += 1

    pool_exchange_id = f"EX_{PROT_POOL_ID}"
    pool_ub = (
        model.reactions.get_by_id(pool_exchange_id).upper_bound
        if pool_exchange_id in {r.id for r in model.reactions}
        else 0
    )
    logger.info(
        "Full GECKO applied: %d reactions constrained, %d draw reactions, "
        "protein pool ub = %.6g.",
        n_enzyme_constraints,
        len(builder.draw_reaction_ids),
        pool_ub,
    )

    return GECKOFullAnalysis.from_results(
        log={
            "sigma": sigma,
            "ptot": ptot,
            "f_factor": f_factor,
            "n_enzyme_constraints": n_enzyme_constraints,
        },
        ec_model=model,
        protein_pool_id=PROT_POOL_ID,
        draw_reactions=builder.draw_reaction_ids,
        arm_reactions=builder.arm_reaction_ids,
    )
