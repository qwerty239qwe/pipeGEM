from pathlib import Path
from functools import wraps
from typing import Union, Dict

import pandas as pd
import cobra
from cobra.flux_analysis.variability import flux_variability_analysis
from cobra.flux_analysis.parsimonious import pfba
from cobra.sampling import sampling

from pipeGEM.integration.constraints import add_constraint, add_constraint_f, constraint_dict, post_process


ANALYSIS_METHODS = {}


def register_analysis(name: str):
    def register(func):
        assert name not in ANALYSIS_METHODS, "Name collision: " + name
        ANALYSIS_METHODS[name] = func

        @wraps(func)
        def do_work(*args, **kwargs):
            return func(*args, **kwargs)
        return do_work
    return register


@register_analysis("FBA")
def _fba(model, **kwargs):
    return model.optimize(**kwargs)


@register_analysis("FVA")
def _fva(model, is_loopless=True, fraction_of_optimum=0.8, **kwargs):
    return flux_variability_analysis(model, loopless=is_loopless, fraction_of_optimum=fraction_of_optimum, **kwargs)


@register_analysis("pFBA")
def _pfba(model, fraction_of_optimum=1.0, **kwargs):
    return pfba(model, fraction_of_optimum=fraction_of_optimum)


@register_analysis("sampling")
def _sampling(model, obj_lb_ratio=0.75, n=5000, **kwargs):
    obj_lb = model.slim_optimize() * obj_lb_ratio
    with model:
        biom = model.problem.Constraint(model.objective.expression, obj_lb)
        model.solver.add(biom)
        return sampling.sample(model, n=n, **kwargs)


@add_constraint_f
def flux_analysis(model, method, **kwargs):
    assert method in ANALYSIS_METHODS, "The method must in " + ",".join([m for m in ANALYSIS_METHODS])
    return ANALYSIS_METHODS[method](model, **kwargs)


class FluxAnalyzer:
    def __init__(self,
                 model: cobra.Model,
                 rxn_expr_score=None,
                 solver: str = 'gurobi'):
        self.method_dicts = {'FBA': getattr(self, '_fba'),
                             'FVA': getattr(self, '_fva'),
                             'pFBA': getattr(self, '_pfba'),
                             'sampling': getattr(self, '_sampling')}
        self.model = model
        self.model.solver = solver
        if not rxn_expr_score:
            self.rxn_expr_score = rxn_expr_score
        elif isinstance(rxn_expr_score, dict):
            self.rxn_expr_score = rxn_expr_score
        else:
            self.rxn_expr_score = rxn_expr_score.rxn_scores
        self._df = {constr: {name: None
                             for name, method in self.method_dicts.items()} for constr, _ in constraint_dict.items()}

    @add_constraint
    def do_analysis(self,
                    method: str,
                    constr,
                    postprocess_kwargs=None,
                    **kwargs) -> None:
        if postprocess_kwargs is None:
            postprocess_kwargs = {}
        if method not in self.method_dicts:
            raise ValueError('This method is not available, please choose one of the method below: '
                             '\n [FBA, FVA, pFBA, sampling]')
        self._df[constr][method] = post_process(self.method_dicts[method](model=self.model,
                                                                          **kwargs),
                                                constr=constr,
                                                **postprocess_kwargs)

    def get_flux(self,
                 method: str,
                 constr: str = "default",
                 keep_rc: bool = False) -> pd.DataFrame:
        """
        Get stored flux analysis result. If the result is not found, do the flux analysis and return the result.

        Parameters
        ----------
        method: str
            The flux analysis method
        constr: str
            Applied additional constraint type
        keep_rc: bool
            If the returned dataframe contains reduced costs column
        Returns
        -------
        flux_df: pd.DataFrame
            Flux result stored in a pd.DataFrame,
            expected rows: reactions
            expected columns: depends on the method,
                FBA and pFBA: [fluxes, [reduced_costs]],
                FVA: [maximum, minimum],
                samping: [0, 1, ..., n] (n = number of samples)
        """
        if self._df[constr][method] is None:
            self.do_analysis(methods=method, const=constr)
        df = self._df[constr][method].to_frame() \
            if not isinstance(self._df[constr][method], pd.DataFrame) else self._df[constr][method]
        return df if keep_rc else df.drop(columns=["reduced_costs"])

    def get_sol(self, method=None, constr="default"):
        if method not in ["pFBA", "FBA"]:
            raise AttributeError("This method doesn't have cobra.Solution, use get_flux instead")

        if method is not None:
            if not isinstance(self._df[constr][method], cobra.Solution):
                self.do_analysis(methods=method, const=constr)
            return self._df[constr][method]
        return self._df[constr]

    def save_analysis(self, file_path):
        root_path = Path(file_path) if isinstance(file_path, str) else file_path
        for constr, method_df_dic in self._df.items():
            saved_df_path = (root_path / Path(constr.value))
            saved_df_path.mkdir(parents=True, exist_ok=True)
            for method, df_dic in method_df_dic.items():
                if df_dic is not None:
                    df_dic.to_csv((saved_df_path / Path(method)).with_suffix(".tsv"), sep='\t')

    def load_analysis(self, file_path):
        root_path = Path(file_path) if isinstance(file_path, str) else file_path
        for constr, method_df_dic in self._df.items():
            saved_df_path = (root_path / Path(constr.value))
            for method, _ in method_df_dic.items():
                if (saved_df_path / Path(method)).exists():
                    method_df_dic[method] = pd.read_csv((saved_df_path / Path(method)).with_suffix(".tsv"),
                                                        sep='\t', index_col=0)

    @staticmethod
    def _fba(model, **kwargs):
        return model.optimize(**kwargs)

    @staticmethod
    def _fva(model, is_loopless=True, fraction_of_optimum=0.8, **kwargs):
        return flux_variability_analysis(model, loopless=is_loopless, fraction_of_optimum=fraction_of_optimum, **kwargs)

    @staticmethod
    def _pfba(model, fraction_of_optimum=1.0, **kwargs):
        return pfba(model, fraction_of_optimum=fraction_of_optimum)

    @staticmethod
    def _sampling(model, obj_lb_ratio=0.75, n=5000, **kwargs):
        obj_lb = model.slim_optimize() * obj_lb_ratio
        with model:
            biom = model.problem.Constraint(model.objective.expression, obj_lb)
            model.solver.add(biom)
            return sampling.sample(model, n=n, **kwargs).T