from typing import Union, List

import pandas as pd
import cobra
import numpy as np
from numpy import linalg as LA
from optlang.symbolics import Zero

from pipeGEM.core import Problem


def LP3(J: Union[set, np.ndarray, List[str]],
        model: cobra.Model,
        epsilon: float) -> list:
    """

    Parameters
    ----------
    J: Union[set, np.ndarray, List[str]]
        A set (or list, array) of rxn IDs
    model: cobra.Model
        The model contains J
    epsilon
        Threshold of non-zero fluxes
    Returns
    -------
        A list of rxn IDs that have zero flux while J are the objective function.
    """
    with model:
        if isinstance(J, set) or isinstance(J, np.ndarray) or isinstance(J, list):
            model.objective = [model.reactions.get_by_id(j) for j in J]
        else:
            model.objective = model.reactions.get_by_id(J)
        fm = model.optimize().to_frame()["fluxes"].abs()
    return fm[fm > 0.99*epsilon].index.to_list()


def non_convex_LP3(J, model, epsilon) -> list:
    # check both directions
    if isinstance(J, set) or isinstance(J, np.ndarray) or isinstance(J, list):
        assert len(J) == 1
        obj_rxn = model.reactions.get_by_id(next(iter(J)))
    else:
        obj_rxn = model.reactions.get_by_id(J)
    with model:
        model.objective = obj_rxn
        if not obj_rxn.reversibility:
            if obj_rxn.lower_bound >= 0:
                fm = model.optimize(objective_sense="maximize").to_frame()["fluxes"].abs()
            elif obj_rxn.upper_bound <= 0:
                fm = model.optimize(objective_sense="minimize").to_frame()["fluxes"].abs()
            else:
                fm = pd.Series([0 for _ in model.reactions])
        else:
            fm = model.optimize(objective_sense="maximize").to_frame()["fluxes"].abs()
            if fm.loc[obj_rxn.id] < epsilon:
                fm = model.optimize(objective_sense="minimize").to_frame()["fluxes"].abs()
                print(fm.loc[obj_rxn.id], obj_rxn.id)
                if fm.loc[obj_rxn.id] < epsilon:
                    fm = pd.Series([0 for _ in model.reactions])

    return fm[fm > 0.99*epsilon].index.to_list()


# def non_convex_compute_obj(v: np.array, rho: np.array, epsilon: float):
#     return np.minimum(abs(v)/epsilon, np.ones(v.shape)).dot(rho)


def non_convex_LP7(J, model: cobra.Model, epsilon: float, use_abs=True) -> list:
    max_iter = 20
    with model:
        prob = model.problem
        vars = []
        consts = []
        for j in J:
            rxn = model.reactions.get_by_id(j)
            var = prob.Variable(f"LP7_z_{rxn.id}", lb=1, ub=max(1,
                                                                abs(rxn.lower_bound / epsilon),
                                                                abs(rxn.upper_bound / epsilon)))  # 0 <= zi <= eps
            const_1 = prob.Constraint(rxn.flux_expression / epsilon - 1.0 * var,
                                      name=f"const_1_LP7_{rxn.id}", ub=0)  # vi >= zi
            const_2 = prob.Constraint(-rxn.flux_expression / epsilon - 1.0 * var,
                                      name=f"const_2_LP7_{rxn.id}", ub=0)  # -vi >= zi
            consts.extend([const_1, const_2])
            vars.append(var)
        model.add_cons_vars(vars + consts)
        jcoef = {j: 1 for j in J}
        obj_old, v_old = len(J), pd.Series([0 for _ in model.reactions], index=[r.id for r in model.reactions])
        v_old.loc[J] = 1
        for i in range(max_iter):
            model.objective = prob.Objective(Zero, sloppy=True)
            model.objective = {model.reactions.get_by_id(j): -c/epsilon for j, c in jcoef.items()}
            model.objective.set_linear_coefficients({v: 1 for v in vars})
            sol = model.optimize(objective_sense="minimize", raise_error=False)
            if sol.status == "infeasible":
                sol = pd.Series([0 for _ in model.reactions], name="fluxes")
                break
            signed_sol = sol.to_frame()["fluxes"].apply(np.sign)
            jcoef = {k: v for k, v in signed_sol.items() if v != 0}
            obj_new = np.sum(np.minimum((sol.to_frame()["fluxes"].loc[J] / 1e-6).abs(), np.ones(len(J))))
            v_new = sol.to_frame()["fluxes"]
            # print(obj_new, obj_old)
            if abs(obj_new - obj_old) < epsilon or LA.norm(v_new - v_old) < epsilon:
                break
            else:
                obj_old, v_old = obj_new, v_new
    if use_abs:
        fm = sol.to_frame()["fluxes"].abs()
    else:
        fm = sol.to_frame()["fluxes"]

    return fm[fm > 0.99*epsilon].index.to_list()


def LP7(J,
        model: cobra.Model,
        epsilon: float,
        use_abs=True) -> list:
    """
    LP7 tries to maximize the number of feasible fluxes in J whose value is at least epsilon (Nikos Vlassis, et al. 2013)

    Parameters
    ----------
    J
        A rxn set that the number of feasible fluxes in it is maximized by LP7
    model
        Used cobra model that contains all of the J
    epsilon
        Threshold of non-zero fluxes
    use_abs
        the returned reactions are selected by its absolute value or not
    Returns
    -------
        a list of rxn IDs, the rxns can produce feasible fluxes.
    """
    with model:
        prob = model.problem
        vars = []
        consts = []
        for j in J:
            rxn = model.reactions.get_by_id(j)
            var = prob.Variable(f"LP7_z_{rxn.id}", lb=0, ub=epsilon)  # 0 <= zi <= eps
            const = prob.Constraint(-rxn.flux_expression + 1.0 * var,
                                    name=f"const_LP7_{rxn.id}", ub=0)  # vi >= zi
            consts.append(const)
            vars.append(var)
        model.add_cons_vars(vars + consts)
        model.objective = prob.Objective(Zero, sloppy=True)
        model.objective.set_linear_coefficients({v: -1 for v in vars})
        sol = model.optimize(objective_sense="minimize", raise_error=True)
    if use_abs:
        fm = sol.to_frame()["fluxes"].abs()
    else:
        fm = sol.to_frame()["fluxes"]

    return fm[fm > 0.99*epsilon].index.to_list()


def LP9(K,
        P,
        NonP,
        model: cobra.Model,
        epsilon) -> list:
    """
    LP9 minimizes the L1 norm of fluxes in the penalty set P, subject to a minimum flux constraint on the set K.
    (Nikos Vlassis, et al. 2013)

    Parameters
    ----------
    K
        A rxn set that LP9 maintains their flux feasibility (flux value in K must be larger than the epsilon)
    P
        A rxn set that LP9 tries to reduce the number of feasible reactions in it.
    NonP
        A rxn set that LP9 ignores (neither maintain nor penalize their flux feasibility)
    model
        Used cobra model in this algo. K, P and NonP should be contained in the model
    epsilon
        Threshold of non-zero fluxes
    Returns
    -------
        The result rxn ids list
    """
    scaling_factor = 1e5
    with model:
        prob = model.problem
        vars = []
        objs = []
        consts = []
        # print(f"In LP9, |P| = {len(P)}, |K| = {len(K)}")

        for p in P:  # penalized rxns
            rxn = model.reactions.get_by_id(p)
            z = prob.Variable(f"LP9_z_{rxn.id}", lb=0, ub=max(abs(rxn.lower_bound),
                                                              abs(rxn.upper_bound)) * scaling_factor)  # define z
            fconst = prob.Constraint(-z + rxn.flux_expression,
                                     name=f"const_forward_LP9_{rxn.id}", ub=0)  # vi <= zi
            rconst = prob.Constraint(-z - rxn.flux_expression,
                                     name=f"const_reverse_LP9_{rxn.id}", ub=0)  # -zi <= vi
            vars.append(z)
            if p not in NonP:
                objs.append(z)
            consts.extend([fconst, rconst])
        for k in K:  # to maintain
            rxn = model.reactions.get_by_id(k)

            # vi >= eps (for all i in K)
            kconst = prob.Constraint(-rxn.flux_expression,
                                     name=f"const_K_LP9_{rxn.id}", ub=-epsilon * scaling_factor)
            consts.append(kconst)
        for r in model.reactions:
            r.lower_bound, r.upper_bound = r.lower_bound * scaling_factor, r.upper_bound * scaling_factor

        model.add_cons_vars(vars + consts)
        model.objective = prob.Objective(Zero, sloppy=True)
        model.objective.set_linear_coefficients({v: 1.0 for v in objs})  # sum of zi
        # print(len(model.constraints), len(model.variables))
        try:
            sol = model.optimize(objective_sense="minimize", raise_error=True)
        except:
            print("infeasible result: K = ", K)
            return []
        fm = sol.to_frame()["fluxes"].abs()
    return fm[fm > 0.99 * epsilon].index.to_list()  # supp


class BlockedProblem(Problem):
    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)

    def _transpose_S(self):
        m, n = self.S.shape
        self._irrev = ~self.get_rev()
        self.S = self.S.T
        self.objs = np.zeros((m,))
        self.lbs = np.array([-np.inf for _ in range(m)])
        self.ubs = np.array([np.inf for _ in range(m)])
        self.b = np.zeros((n,))
        self.c = np.array(["E" for _ in range(n)])
        self.c[self._irrev] = "L"
        self.v = np.array(["C" for _ in range(m)])
        self.col_names, self.row_names = self.row_names, self.col_names
        self._check_matrix()

    def modify_problem(self):
        m, n = self.S.shape
        self._transpose_S()

        # extend right
        ext_objs = np.zeros((n,))
        ext_objs[self._irrev] = 1
        ext_S = -np.eye(n)
        e_lbs = np.zeros((n,))
        e_ubs = np.zeros((n,))
        e_lbs[self._irrev] = -1
        e_v = np.array(["C" for _ in range(n)])
        names = [f"rxn_{i}" for i in range(n)]
        self.extend_horizontal(ext_S, e_v, e_lbs, e_ubs, ext_objs, names)

