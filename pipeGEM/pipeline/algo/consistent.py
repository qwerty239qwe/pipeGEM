import cobra

from pipeGEM import Model
from pipeGEM.pipeline import Pipeline
from pipeGEM.integration.algo.fastcore import fastcc
from pipeGEM.integration.algo.swiftcore import swiftcc


class FastCC(Pipeline):
    def __init__(self):
        super().__init__()

    def run(self, model, *args, **kwargs):
        self.output = fastcc(model, **kwargs)
        return self.output


class SwiftCC(Pipeline):
    def __init__(self):
        super().__init__()

    def run(self, model, return_model=True, *args, **kwargs):
        if isinstance(model, cobra.Model):
            model = Model(model, name_tag="swiftcc")
        self.output = swiftcc(model, return_model=return_model, **kwargs)
        return self.output
