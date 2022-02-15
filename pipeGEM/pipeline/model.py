from ._base import Pipeline
import logging
from pipeGEM.integration.utils import apply_medium_constraint


class MediumConstraint(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log = logging.getLogger((self._prev_lvl_pl_name + ".") if self._prev_lvl_pl_name is not None else "" +
                                      type(self).__name__)
        self._log.debug("Init MediumConstraint pipeline.")

    def run(self, model, medium, except_rxns, *args, **kwargs):
        constr_rxns = apply_medium_constraint(model, medium, except_rxns=except_rxns)
        self._log.info(f"{len(constr_rxns)} reactions are applied new constraints.")
        self._log.debug(f"Medium constraints are applied to: {', '.join(constr_rxns)}")
