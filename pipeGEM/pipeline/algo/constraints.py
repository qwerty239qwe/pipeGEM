import cobra
import tqdm

from pipeGEM.pipeline import Pipeline
from pipeGEM.pipeline.preprocessing import GeneDataSet
from pipeGEM.integration import constraints
from pipeGEM.analysis import modified_pfba


class BaseConstrPipeline(Pipeline):
    def __init__(self, data_df, get_model, **kwargs):
        super().__init__(**kwargs)
        self.gene_dataset = GeneDataSet(data_df)
        self.data_df = data_df
        self.get_model = get_model

    def run(self, model, *args, **kwargs):
        raise NotImplementedError()


class GIMME(BaseConstrPipeline):
    def __init__(self, data_df, **kwargs):
        super().__init__(data_df, **kwargs)

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


class Eflux(BaseConstrPipeline):
    def __init__(self, data_df, **kwargs):
        super().__init__(data_df, **kwargs)

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


class RIPTiDe(BaseConstrPipeline):
    def __init__(self, data_df, **kwargs):
        super().__init__(data_df, **kwargs)

    def run(self, model: cobra.Model, score_threshold=0.8,
            ignored_reactions=None,
            *args, **kwargs):
        expr_dict = self.gene_dataset(model)

        self.output = {}
        for c in tqdm.tqdm(self.data_df.columns):

            if self.get_model:
                const_model = model.copy()
                constraints.RIPTiDe(const_model,
                                    rxn_expr_score=expr_dict[c].rxn_scores,
                                    **kwargs)
                self.output[c] = const_model
            else:
                with model:
                    obj_dic = {r.id: r.objective_coefficient for r in model.reactions if r.objective_coefficient > 0}
                    result = constraints.RIPTiDe(model,
                                                 rxn_expr_score=expr_dict[c].rxn_scores,
                                                 **kwargs)
                    if "return_max_score_flux" in kwargs and kwargs["return_max_score_flux"]:
                        self.output[c] = constraints.post_process(result, constr="GIMME")
                        continue

                    sol = model.optimize(objective_sense="maximize")
                    con = model.problem.Constraint(model.objective.expression, lb=sol.objective_value * 0.8)
                    model.add_cons_vars([con])
                    model.objective = {model.reactions.get_by_id(r): v for r, v in obj_dic.items()}
                    sol = modified_pfba(model, ignored_reactions=ignored_reactions)
                    sol_df = constraints.post_process(sol, constr="GIMME")
                    self.output[c] = sol_df
        return self.output

