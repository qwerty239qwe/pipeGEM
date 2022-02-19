from ._base import Pipeline
from pipeGEM.integration.utils import apply_medium_constraint


class MediumConstraint(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, model, medium, except_rxns, *args, **kwargs):
        constr_rxns = apply_medium_constraint(model, medium, except_rxns=except_rxns)
        self._info(f"{len(constr_rxns)} reactions are applied new constraints.")
        self._debug(f"Medium constraints are applied to: {', '.join(constr_rxns)}")
