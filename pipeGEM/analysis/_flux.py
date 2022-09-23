from pathlib import Path
from functools import wraps
from typing import Union, Dict
from itertools import chain

import pandas as pd
import numpy as np
import cobra
from cobra.core.solution import get_solution
from cobra.flux_analysis.variability import flux_variability_analysis
from cobra.flux_analysis.parsimonious import pfba
from cobra.sampling import sampling
from cobra.util.solver import fix_objective_as_constraint
from optlang.symbolics import Zero

from .._constant import var_type_dict
from .results import *
from pipeGEM.analysis._gapsplit import gapsplit
from pipeGEM.utils import ObjectFactory


class FluxAnalyzers(ObjectFactory):
    def __init__(self):
        super().__init__()


class FluxAnalyzer:
    def __init__(self,
                 model,
                 analysis_obj: "FluxAnalysis",
                 solver = "glpk"):
        self.model = model
        self.model.solver = solver
        self._solver_name = solver
        self._analysis_obj = analysis_obj

    @property
    def solver_name(self):
        return self._solver_name

    def analysis_func(self, **kwargs):
        raise NotImplementedError

    def analyze(self, **kwargs) -> "FluxAnalysis":
        self._analysis_obj.add_result(self.analysis_func(**kwargs))
        return self._analysis_obj


class FBA_Analyzer(FluxAnalyzer):
    def __init__(self, model, solver, log=None):
        super().__init__(model=model,
                         solver=solver,
                         analysis_obj=FBA_Analysis(log=log))

    def analysis_func(self, **kwargs):
        return self.model.optimize(**kwargs)


class pFBA_Analyzer(FBA_Analyzer):
    def __init__(self, model, solver, log=None):
        super().__init__(model=model,
                         solver=solver,
                         log=log)

    def analysis_func(self, fraction_of_optimum=1.0, **kwargs):
        return pfba(model=self.model,
                    fraction_of_optimum=fraction_of_optimum,
                    **kwargs)


class FVA_Analyzer(FluxAnalyzer):
    def __init__(self, model, solver, log=None):
        super().__init__(model=model,
                         solver=solver,
                         analysis_obj=FVA_Analysis(log=log))

    def analysis_func(self,
                      is_loopless=True,
                      fraction_of_optimum=0,
                      **kwargs):
        return flux_variability_analysis(self.model,
                                         loopless=is_loopless,
                                         fraction_of_optimum=fraction_of_optimum,
                                         **kwargs)


class SamplingAnalyzer(FluxAnalyzer):
    def __init__(self, model, solver, log=None):
        super().__init__(model=model,
                         solver=solver,
                         analysis_obj=SamplingAnalysis(log=log))

    def analysis_func(self,
                      obj_lb_ratio=0.75,
                      n=5000,
                      **kwargs):
        obj_lb = self.model.slim_optimize() * obj_lb_ratio
        with self.model:
            biom = self.model.problem.Constraint(self.model.objective.expression, obj_lb)
            self.model.solver.add(biom)
            if kwargs.get("method") == "gapsplit":
                kwargs["gurobi_direct"] = (self.solver_name == "gurobi")
                kwargs.pop("method")
                return gapsplit(self.model, n=n, **kwargs)
            else:
                method = kwargs.pop("method")

            return sampling.sample(self.model, n=n, **kwargs)


flux_analyzers = FluxAnalyzers()
flux_analyzers.register("FBA", FBA_Analyzer)
flux_analyzers.register("FVA", FVA_Analyzer)
flux_analyzers.register("pFBA", pFBA_Analyzer)
flux_analyzers.register("sampling", SamplingAnalyzer)


class ProblemAnalyzer:
    def __init__(self,
                 problem,
                 solver: str = None):
        self.S, self.v, self.v_lbs, self.v_ubs, self.b, self.csense = problem.S, problem.v, \
                                                                      problem.lbs, problem.ubs, problem.b, problem.c
        self.objs, self.col_names, self.row_names = problem.objs, problem.col_names, problem.row_names
        self.anchor_n = problem.anchor_n
        self.new_model = cobra.Model()
        if solver is not None:
            self.new_model.solver = solver
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

    def get_fluxes(self, direction="max", raise_error=True) -> pd.DataFrame:
        self.new_model.solver.objective.direction = direction
        self.new_model.solver.optimize()
        print(self.new_model.solver.status)
        if raise_error and self.new_model.solver.status != "optimal":
            raise RuntimeError("The solver's status is infeasible")
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


def add_mod_pfba(
        model,
        objective = None,
        fraction_of_optimum: float = 1.0,
        reactions = None,
        weights = None,
        direction = "min"
    ):
    if objective is not None:
        model.objective = objective
    if model.solver.objective.name == "_pfba_objective":
        raise ValueError("The model already has a pFBA objective.")
    if fraction_of_optimum != 0:
        fix_objective_as_constraint(model, fraction=fraction_of_optimum)
    reactions = model.reactions if reactions is None else reactions
    if weights is None:
        reaction_var_dict = {rxn.forward_variable: 1 for rxn in reactions}
        reaction_var_dict.update({rxn.reverse_variable: 1 for rxn in reactions})
    else:
        if any([np.isnan(x) for x in weights.values()]):
            invalid = [k for k, v in weights.items() if np.isnan(v)]
            print(f"NaN detected: {invalid}")
            raise ValueError("weights dict contains NaN")
        rxn_ids = [r.id for r in reactions]
        weights = {k: v for k, v in weights.items() if k in rxn_ids}
        reaction_var_dict = {model.reactions.get_by_id(k).forward_variable: v for k, v in weights.items()}
        reaction_var_dict.update({model.reactions.get_by_id(k).reverse_variable: v for k, v in weights.items()})

    # model.objective = model.problem.Objective(
    #     Zero, direction=direction, sloppy=True, name="_pfba_objective"
    # )
    model.objective.set_linear_coefficients({v: 0 for v in model.variables})
    model.objective.set_linear_coefficients({k: v for k, v in reaction_var_dict.items()})


def _add_max_abs_flux(
        model,
        objective = None,
        fraction_of_optimum: float = 1.0,
        reactions = None,
        weights = None,
    ):
    pass


def modified_pfba(model,
                  ignored_reactions=None,
                  fraction_of_optimum: float = 1.0,
                  objective= None,
                  weights = None,
                  direction = "min"
                  ):
    reactions = (
        model.reactions if ignored_reactions is None else [r for r in model.reactions if r.id not in ignored_reactions]
    )
    with model as m:
        add_mod_pfba(m,
                     objective=objective,
                     reactions=reactions,
                     fraction_of_optimum=fraction_of_optimum,
                     weights=weights,
                     direction=direction)
        m.slim_optimize(error_value=None)
        solution = get_solution(m, reactions=reactions)
    return solution


def add_norm_constraint(model,
                        ignored_reactions=None,
                        coef_dict=None,
                        degree=1,
                        ub=1,
                        name="norm_constraint_for_reactions"):
    import gurobipy
    if coef_dict is None:
        coef_dict = {r.id: 1 for r in model.reactions}
    terms = [] if degree != 1 else {}
    ignored_reactions = ignored_reactions if ignored_reactions is not None else []
    for r in model.reactions:
        if r.id not in ignored_reactions:
            if degree == 2:
                f_var = model.solver.problem.getVarByName(r.forward_variable.name)
                r_var = model.solver.problem.getVarByName(r.reverse_variable.name)
                terms.append(coef_dict[r.id] * f_var * f_var)
                terms.append(coef_dict[r.id] * r_var * r_var)
            elif degree == 1:
                terms.update({r.forward_variable: coef_dict[r.id],
                              r.reverse_variable: coef_dict[r.id]})

    if degree == 2:
        lhs = gurobipy.quicksum(terms)
        model.solver.problem.addQConstr(lhs, gurobipy.GRB.LESS_EQUAL, ub ** degree, name=name)
    elif degree == 1:
        norm1_cons = model.problem.Constraint(Zero,
                                              name=name, lb=0, ub=ub)
        model.add_cons_vars([norm1_cons])
        model.solver.update()
        norm1_cons.set_linear_coefficients(terms)
    # model.solver.problem.params.NonConvex = 2  # every norm is a convex function
    model.solver.update()


def max_abs_flux(model, ):
    pass