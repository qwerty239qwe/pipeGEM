from pathlib import Path
from functools import wraps
from typing import Union, Dict

import pandas as pd
import numpy as np
import cobra
from cobra.flux_analysis.variability import flux_variability_analysis
from cobra.flux_analysis.parsimonious import pfba
from cobra.sampling import sampling
from optlang.symbolics import Zero

from pipeGEM.integration.constraints import add_constraint, add_constraint_f, constraint_dict, post_process
from .._constant import var_type_dict


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
                             for name, method in self.method_dicts.items()}
                    for constr, _ in constraint_dict.items()}

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
                 keep_rc: bool = False,
                 **kwargs) -> pd.DataFrame:
        """
        Get stored flux analysis result.
        If the result is not found, do the flux analysis and return the result.

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
            self.do_analysis(method=method, constr=constr, **kwargs)
        df = self._df[constr][method].to_frame() \
            if not isinstance(self._df[constr][method], pd.DataFrame) else self._df[constr][method]
        return df if keep_rc or "reduced_costs" not in df.columns else df.drop(columns=["reduced_costs"])

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


class ProblemAnalyzer:
    def __init__(self,
                 problem,
                 solver: str = 'gurobi'):
        self.S, self.v, self.v_lbs, self.v_ubs, self.b, self.csense = problem.S, problem.v, \
                                                                      problem.lbs, problem.ubs, problem.b, problem.c
        self.objs, self.col_names, self.row_names = problem.objs, problem.col_names, problem.row_names
        self.anchor_n = problem.anchor_n
        self.new_model = cobra.Model()
        self.prob = self.new_model.problem
        self.variables = []
        self.r_variables = []
        self.constraints = []
        if self.v is None:
            self.v = ["C" for _ in range(self.S.shape[1])]
        self.setup_constrs()
        self.setup_variables()


    def setup_constrs(self):
        for i, (b_bound, b_type) in enumerate(zip(self.b, self.csense)):
            constr_data = {"name": f"const_{i}" if self.row_names is None else self.row_names[i]}
            if b_type in ("E", "G"):
                constr_data.update({"lb": b_bound})
            if b_type in ("E", "L"):
                constr_data.update({"ub": b_bound})
            n = self.prob.Constraint(
                Zero,
                **constr_data
            )
            self.constraints.append(n)
        self.new_model.solver.add(self.constraints, sloppy=True)
        self.reverse_dic = {v.name: r.name for v, r in zip(self.variables, self.r_variables)}

    def setup_variables(self):
        for i, (v_lb, v_ub, v_type) in enumerate(zip(self.v_lbs, self.v_ubs, self.v)):
            added_vars = []
            if i < self.anchor_n:
                if v_lb > 0:
                    a_lb_f, a_ub_f = None if np.isinf(v_lb) else v_lb, None if np.isinf(v_ub) else v_ub
                    a_lb_r, a_ub_r = 0, 0
                elif v_ub < 0:
                    a_lb_f, a_ub_f = 0, 0
                    a_lb_r, a_ub_r = None if np.isinf(v_ub) else -v_ub, None if np.isinf(v_lb) else -v_lb
                else:
                    a_lb_f, a_ub_f = 0, None if np.isinf(v_ub) else v_ub
                    a_lb_r, a_ub_r = 0, None if np.isinf(v_lb) else -v_lb
                m_r = self.prob.Variable(name=f"var_{i}_r" if self.col_names is None else self.col_names[i] + "_r",
                                         type=var_type_dict[v_type],
                                         lb=a_lb_r,
                                         ub=a_ub_r)
                added_vars.append(m_r)
            else:
                a_lb_f, a_ub_f = v_lb, v_ub
            m = self.prob.Variable(name=f"var_{i}" if self.col_names is None else self.col_names[i],
                                   type=var_type_dict[v_type],
                                   lb=a_lb_f,
                                   ub=a_ub_f)
            added_vars.append(m)
            self.new_model.solver.add(added_vars, sloppy=True)
            non_zero_idx = np.nonzero(self.S[:, i])[0]
            for j in non_zero_idx:
                self.new_model.solver.constraints[f"const_{j}" if self.row_names is None
                else self.row_names[j]].set_linear_coefficients({
                                                               m: float(self.S[j, i]),
                                                               m_r: -float(self.S[j, i])
                                                           } if i < self.anchor_n else {
                    m: float(self.S[j, i])
                })

            self.variables.append(m)
            if i < self.anchor_n:
                self.r_variables.append(m_r)
                self.reverse_dic[m_r.name] = m.name

        obj_vars = {self.variables[i]: c for i, c in enumerate(self.objs)}
        obj_vars.update({v: -self.objs[i] for i, v in enumerate(self.r_variables)})
        self.new_model.objective = self.prob.Objective(Zero, sloppy=True)
        self.new_model.objective.set_linear_coefficients(obj_vars)
        self.new_model.solver.update()
        print(self.new_model.objective)

    def get_fluxes(self, direction="max") -> pd.DataFrame:
        self.new_model.solver.objective.direction = direction
        self.new_model.solver.optimize()
        print(self.new_model.solver.status)
        print(self.new_model.solver.objective.value)
        vals = {}
        _skips = []
        for var_name, val in self.new_model.solver.primal_values.items():
            if var_name in self.reverse_dic:
                if self.reverse_dic[var_name] in vals:
                    vals[self.reverse_dic[var_name]] -= val
                else:
                    vals[self.reverse_dic[var_name]] = -val
            else:
                if var_name in vals:
                    vals[var_name] += val
                else:
                    vals[var_name] = val
        return pd.DataFrame({"fluxes": vals}).loc[self.col_names, :]