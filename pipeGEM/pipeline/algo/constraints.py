from pipeGEM.pipeline import Pipeline
from pipeGEM.pipeline.preprocessing import GeneDataSet


class GIMME(Pipeline):
    def __init__(self, data_df):
        super(GIMME, self).__init__()
        self.gene_dataset = GeneDataSet(data_df)

    def run(self, model, dataset):
        raise NotImplementedError()


class Eflux(Pipeline):
    def __init__(self, data_df):
        super(Eflux, self).__init__()
        self.gene_dataset = GeneDataSet(data_df)

    def run(self, model, dataset):
        raise NotImplementedError()