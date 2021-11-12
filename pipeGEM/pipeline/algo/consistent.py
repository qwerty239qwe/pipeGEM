from pipeGEM.pipeline import Pipeline
from pipeGEM.integration.algo.fastcore import fastcc


class FastCC(Pipeline):
    def __init__(self):
        super().__init__()

    def run(self, model, *args, **kwargs):
        self.output = fastcc(model, **kwargs)
        return self.output


class SwiftCC(Pipeline):
    pass