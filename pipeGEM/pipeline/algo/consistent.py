import cobra

from pipeGEM import Model
from pipeGEM.pipeline import Pipeline
from pipeGEM.integration.algo.fastcore import fastcc
from pipeGEM.integration.algo.swiftcore import swiftcc


class FastCC(Pipeline):
    def __init__(self):
        super().__init__()

    def get_log(self, *args, **kwargs):
        n_rxn = len(self.output["model"].reactions)
        n_met = len(self.output["model"].metabolites)
        n_gen = len(self.output["model"].genes)
        obj_v = self.output["model"].slim_optimize()
        log = f"Number of reactions: {n_rxn}\n" \
              f"Number of metabolites: {n_met}\n" \
              f"Number of genes: {n_gen}\n" \
              f"Objective value: {obj_v}"

        if "removed_rxn_ids" in self.output:
            log += f"Number of removed rxns: {len(self.output['removed_rxn_ids'])}"

        return log

    def run(self,
            model,
            tol=1e-6,
            return_model=True,
            return_rxn_ids=True,
            return_removed_rxn_ids=True,
            *args,
            **kwargs):
        self.output = fastcc(model,
                             epsilon=tol,
                             return_model=return_model,
                             return_rxn_ids=return_rxn_ids,
                             return_removed_rxn_ids=return_removed_rxn_ids, **kwargs)
        return self.output


class SwiftCC(Pipeline):
    def __init__(self):
        super().__init__()

    def run(self, model, return_model=True, *args, **kwargs):
        if isinstance(model, cobra.Model):
            model = Model(model, name_tag="swiftcc")
        self.output = swiftcc(model, return_model=return_model, **kwargs)
        return self.output
