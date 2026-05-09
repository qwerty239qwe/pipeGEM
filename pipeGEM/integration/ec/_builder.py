"""Helper class for building enzyme-constrained model structures (GECKO)."""
import cobra
import numpy as np

from pipeGEM._logging import get_logger
from pipeGEM.utils._manipulate import make_irrev_rxn

logger = get_logger(__name__)

PROT_POOL_ID = "prot_pool"


class ECModelBuilder:
    """Constructs the structural modifications for a full GECKO ecModel.

    The GECKO formulation adds:

    1. A **protein pool** pseudo-metabolite whose total availability is
       bounded by ``ptot * f_factor * sigma``.
    2. **Draw reactions** that convert protein pool into individual enzyme
       pseudo-metabolites.
    3. **Arm reactions** that replace the original metabolic reactions and
       include an enzyme pseudo-metabolite as a substrate consumed with
       stoichiometry ``MW / kcat``.
    """

    def __init__(self, sigma: float = 0.5, ptot: float = 0.5, f_factor: float = 0.5):
        self.sigma = sigma
        self.ptot = ptot
        self.f_factor = f_factor
        self._draw_rxn_ids: list = []
        self._arm_rxn_ids: list = []

    # ------------------------------------------------------------------
    # Protein pool
    # ------------------------------------------------------------------

    def add_protein_pool(self, model: cobra.Model) -> cobra.Metabolite:
        """Add the protein pool pseudo-metabolite and its exchange reaction."""
        if PROT_POOL_ID in {met.id for met in model.metabolites}:
            prot_pool = model.metabolites.get_by_id(PROT_POOL_ID)
        else:
            prot_pool = cobra.Metabolite(
                PROT_POOL_ID,
                name="Protein pool",
                compartment="c",
            )

        pool_exchange_id = f"EX_{PROT_POOL_ID}"
        pool_ub = self.ptot * self.f_factor * self.sigma
        if pool_exchange_id in {rxn.id for rxn in model.reactions}:
            pool_exchange = model.reactions.get_by_id(pool_exchange_id)
            if prot_pool not in pool_exchange.metabolites:
                raise ValueError(
                    f"Existing {pool_exchange_id} does not exchange {PROT_POOL_ID}."
                )
            pool_exchange.lower_bound = 0.0
            pool_exchange.upper_bound = pool_ub
        else:
            pool_exchange = cobra.Reaction(pool_exchange_id)
            pool_exchange.name = "Protein pool exchange"
            pool_exchange.lower_bound = 0.0
            pool_exchange.upper_bound = pool_ub
            pool_exchange.add_metabolites({prot_pool: 1.0})
            model.add_reactions([pool_exchange])
        logger.debug(
            "Added protein pool: ub = %.6g (ptot=%.4g, f=%.4g, sigma=%.4g)",
            pool_exchange.upper_bound, self.ptot, self.f_factor, self.sigma,
        )
        return prot_pool

    # ------------------------------------------------------------------
    # Draw reactions
    # ------------------------------------------------------------------

    def create_draw_reaction(
        self,
        model: cobra.Model,
        prot_pool: cobra.Metabolite,
        prot_id: str,
        mw: float,
    ) -> cobra.Metabolite:
        """Create a draw reaction from the pool to an individual enzyme.

        Returns the enzyme pseudo-metabolite so it can be referenced by
        arm reactions.
        """
        enz_met_id = f"prot_{prot_id}"
        draw_rxn_id = f"draw_{prot_id}"
        met_ids = {m.id for m in model.metabolites}
        rxn_ids = {r.id for r in model.reactions}

        if enz_met_id in met_ids:
            enz_met = model.metabolites.get_by_id(enz_met_id)
        else:
            enz_met = cobra.Metabolite(enz_met_id, name=f"Enzyme {prot_id}", compartment="c")

        if draw_rxn_id in rxn_ids:
            draw_rxn = model.reactions.get_by_id(draw_rxn_id)
            if prot_pool not in draw_rxn.metabolites or enz_met not in draw_rxn.metabolites:
                raise ValueError(
                    f"Existing {draw_rxn_id} is not a valid draw reaction for {enz_met_id}."
                )
            if draw_rxn.id not in self._draw_rxn_ids:
                self._draw_rxn_ids.append(draw_rxn.id)
            return enz_met

        draw_rxn = cobra.Reaction(draw_rxn_id)
        draw_rxn.name = f"Draw reaction for {prot_id}"
        draw_rxn.lower_bound = 0.0
        draw_rxn.upper_bound = 1000.0
        # prot_pool -> enz_met  (stoich: -MW on pool, +1 on enz)
        draw_rxn.add_metabolites({
            prot_pool: -mw,  # grams consumed from pool
            enz_met: 1.0,
        })
        model.add_reactions([draw_rxn])
        self._draw_rxn_ids.append(draw_rxn.id)
        return enz_met

    # ------------------------------------------------------------------
    # Arm reactions
    # ------------------------------------------------------------------

    def prepare_reaction_for_enzyme_constraint(
        self,
        model: cobra.Model,
        rxn: cobra.Reaction,
    ) -> list[cobra.Reaction]:
        """Return non-negative flux reactions ready for enzyme constraints.

        Enzyme pseudo-metabolites must be consumed for flux in every allowed
        direction. Directly adding the enzyme metabolite to a reaction with a
        negative lower bound would make backward flux produce enzyme instead.
        """
        if rxn.lower_bound < 0:
            return make_irrev_rxn(
                model,
                rxn.id,
                add_inplace=True,
                remove_original=True,
            )
        return [rxn]

    def create_arm_reaction(
        self,
        model: cobra.Model,
        rxn: cobra.Reaction,
        enz_met: cobra.Metabolite,
        kcat: float,
    ) -> None:
        """Modify a reaction to consume the enzyme pseudo-metabolite.

        The enzyme is consumed with stoichiometry ``1 / kcat`` (in 1/h units).
        This effectively constrains flux through the reaction to
        ``kcat * [enzyme]``.
        """
        # kcat is typically in 1/s; convert to 1/h
        kcat_per_h = kcat * 3600.0
        if kcat_per_h <= 0:
            logger.warning("kcat <= 0 for %s, skipping arm reaction.", rxn.id)
            return

        coeff = -1.0 / kcat_per_h  # enzyme consumed per unit flux

        # Add enzyme metabolite to the existing reaction
        rxn.add_metabolites({enz_met: coeff})
        self._arm_rxn_ids.append(rxn.id)

    @property
    def draw_reaction_ids(self):
        return list(self._draw_rxn_ids)

    @property
    def arm_reaction_ids(self):
        return list(self._arm_rxn_ids)
