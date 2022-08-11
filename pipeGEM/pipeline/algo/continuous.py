import cobra
import pandas as pd
import tqdm
from cobra.util import fix_objective_as_constraint

from pipeGEM.pipeline import Pipeline
from pipeGEM.pipeline.preprocessing import GeneDataSet
from pipeGEM.integration import continuous
from pipeGEM.analysis import modified_pfba


class BaseFluxConstrPipeline(Pipeline):
    def __init__(self,
                 data_df: pd.DataFrame,
                 get_model,
                 **kwargs):
        """
        This is the abstract class of flux simulation pipeline

        Parameters
        ----------
        data_df
        get_model
        kwargs
        """
        super().__init__(**kwargs)
        self.gene_dataset = GeneDataSet(data_df)
        self.data_df = data_df
        self.get_model = get_model

    def run(self, model, ignored_reactions, *args, **kwargs):
        expr_dict = self.gene_dataset(model)
        self.output = {}
        for c in tqdm.tqdm(self.data_df.columns):
            if self.get_model:
                output_model = model.copy()
                self.apply_constr(output_model, rxn_scores=expr_dict[c].rxn_scores, **kwargs)
                self.output[c] = output_model
            else:
                with model:
                    self.apply_constr(model, rxn_scores=expr_dict[c].rxn_scores, **kwargs)
                    self.output[c] = self.get_flux_sim(model, ignored_reactions=ignored_reactions, **kwargs)
        return self.output

    @staticmethod
    def apply_constr(model, rxn_scores, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def get_flux_sim(model, **kwargs):
        raise NotImplementedError()


class GIMME(BaseFluxConstrPipeline):
    def __init__(self, data_df, **kwargs):
        super().__init__(data_df, **kwargs)

    @staticmethod
    def apply_constr(model, rxn_scores, **kwargs):
        continuous.GIMME(model,
                         rxn_expr_score=rxn_scores,
                         **kwargs)

    @staticmethod
    def get_flux_sim(model, **kwargs):
        return modified_pfba(model=model, ignored_reactions=kwargs.get("ignored_reactions"))


class Eflux(BaseFluxConstrPipeline):
    def __init__(self, data_df, **kwargs):
        super().__init__(data_df, **kwargs)

    @staticmethod
    def apply_constr(model, rxn_scores, **kwargs):
        continuous.Eflux(model,
                         rxn_expr_score=rxn_scores,
                         **kwargs)

    @staticmethod
    def get_flux_sim(model, **kwargs):
        return modified_pfba(model=model, ignored_reactions=kwargs.get("ignored_reactions"))


class Eflux2(BaseFluxConstrPipeline):
    def __init__(self, data_df, **kwargs):
        super().__init__(data_df, **kwargs)

    @staticmethod
    def apply_constr(model, rxn_scores, **kwargs):
        continuous.Eflux(model,
                         rxn_expr_score=rxn_scores,
                         **kwargs)

    @staticmethod
    def get_flux_sim(model, **kwargs):
        fix_objective_as_constraint(model)
        ignored_reactions = kwargs.get("ignored_reactions")
        ignored_reactions = ignored_reactions if ignored_reactions is not None else []
        obj = sum([r.flux_expression ** 2 for r in model.reactions if r.id not in ignored_reactions])
        model.objective = model.problem.Objective(obj, direction='min')
        return model.optimize()


class RIPTiDe(BaseFluxConstrPipeline):
    def __init__(self, data_df, **kwargs):
        super().__init__(data_df, **kwargs)

    @staticmethod
    def apply_constr(model, rxn_scores, **kwargs):
        continuous.RIPTiDe(model,
                           rxn_expr_score=rxn_scores,
                           **kwargs)

    def run(self,
            model: cobra.Model,
            score_threshold=0.8,
            ignored_reactions=None,
            *args, **kwargs):
        expr_dict = self.gene_dataset(model)

        self.output = {}
        for c in tqdm.tqdm(self.data_df.columns):

            if self.get_model:
                const_model = model.copy()
                continuous.RIPTiDe(const_model,
                                   rxn_expr_score=expr_dict[c].rxn_scores,
                                   **kwargs)
                self.output[c] = const_model
            else:
                with model:
                    obj_dic = {r.id: r.objective_coefficient for r in model.reactions if r.objective_coefficient > 0}
                    result = continuous.RIPTiDe(model,
                                                rxn_expr_score=expr_dict[c].rxn_scores,
                                                **kwargs)
                    if "return_max_score_flux" in kwargs and kwargs["return_max_score_flux"]:
                        self.output[c] = continuous.post_process(result, constr="GIMME")
                        continue

                    sol = model.optimize(objective_sense="maximize")
                    con = model.problem.Constraint(model.objective.expression, lb=sol.objective_value * 0.8)
                    model.add_cons_vars([con])
                    model.objective = {model.reactions.get_by_id(r): v for r, v in obj_dic.items()}
                    sol = modified_pfba(model, ignored_reactions=ignored_reactions)
                    sol_df = continuous.post_process(sol, constr="GIMME")
                    self.output[c] = sol_df
        return self.output

