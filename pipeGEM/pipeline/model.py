from ._base import Pipeline
from pipeGEM.integration.utils import apply_medium_constraint


class MediumConstraint(Pipeline):
    def __init__(self):
        super().__init__()

    def run(self, model, medium, except_rxns, *args, **kwargs):
        apply_medium_constraint(model, medium, except_rxns=except_rxns)