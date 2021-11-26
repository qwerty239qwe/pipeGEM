import cobra

from pipeGEM.pipeline import Pipeline
from pipeGEM.pipeline.preprocessing import GeneDataSet
from pipeGEM.integration import constraints


class GIMME(Pipeline):
    def __init__(self, data_df):
        super(GIMME, self).__init__()
        self.gene_dataset = GeneDataSet(data_df)
        self.data_df = data_df

    def run(self, model, **kwargs):
        expr_dict = self.gene_dataset(model)

        self.output = {}
        for c in self.data_df.columns:
            const_model = model.copy()
            constraints.GIMME(const_model,
                              rxn_expr_score=expr_dict[c].rxn_scores,
                              **kwargs)
            self.output[c] = const_model
        return self.output


class Eflux(Pipeline):
    def __init__(self, data_df):
        super().__init__()
        self.gene_dataset = GeneDataSet(data_df)
        self.data_df = data_df

    def run(self, model: cobra.Model, *args, **kwargs):
        expr_dict = self.gene_dataset(model)

        self.output = {}
        for c in self.data_df.columns:
            const_model = model.copy()
            constraints.Eflux(const_model,
                              rxn_expr_score=expr_dict[c].rxn_scores,
                              **kwargs)
            self.output[c] = const_model
        return self.output