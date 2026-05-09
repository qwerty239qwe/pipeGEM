"""Full GECKO: enzyme-constrained model with protein pool.

Creates a full ecModel by adding a protein pool pseudo-metabolite, draw
reactions from the pool to individual enzymes, and modifying metabolic
reactions to consume the enzyme pseudo-metabolite proportional to
``1 / kcat``.
"""
import numpy as np

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
        Protein abundance data (currently used for logging only; the pool
        constraint implicitly limits usage).
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
        if prot_id is None:
            prot_id = rxn_id  # fallback

        rxn = model.reactions.get_by_id(rxn_id)
        constrained_rxns = builder.prepare_reaction_for_enzyme_constraint(model, rxn)
        if not constrained_rxns:
            logger.debug("Reaction %s has no feasible flux direction, skipping.", rxn_id)
            continue

        # Create draw reaction (pool -> individual enzyme)
        enz_met = builder.create_draw_reaction(model, prot_pool, prot_id, mw)

        # Modify each non-negative directional reaction to consume the enzyme
        for constrained_rxn in constrained_rxns:
            builder.create_arm_reaction(model, constrained_rxn, enz_met, kcat)
            n_enzyme_constraints += 1

    logger.info(
        "Full GECKO applied: %d reactions constrained, %d draw reactions, "
        "protein pool ub = %.6g.",
        n_enzyme_constraints,
        len(builder.draw_reaction_ids),
        next(iter(prot_pool.reactions)).upper_bound if prot_pool.reactions else 0,
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
