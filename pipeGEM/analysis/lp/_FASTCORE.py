from typing import Union, List
from warnings import warn

import pandas as pd
import cobra
import numpy as np
from numpy import linalg as LA
from optlang.symbolics import Zero


def LP3(J: Union[set, np.ndarray, List[str]],
        model: cobra.Model,
        epsilon: float,
        flux_logger=None) -> list:
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
        fm = model.optimize(objective_sense="maximize").to_frame()["fluxes"].abs().sort_index()
        if flux_logger is not None:
            rxn_name = J if isinstance(J, str) else list(J)[0]
            flux_logger.add(name=f"LP3_{rxn_name}", flux_series=fm)
    if isinstance(epsilon, pd.Series):
        epsilon = epsilon[fm.index]
    return fm[fm > 0.99*epsilon].index.to_list()


def non_convex_LP3(J, model, epsilon,
                   flux_logger=None) -> list:
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
                # print(fm.loc[obj_rxn.id], obj_rxn.id)
                if fm.loc[obj_rxn.id] < epsilon:
                    fm = pd.Series([0 for _ in model.reactions])
    if flux_logger is not None:
        flux_logger.add(name="nc_LP3", flux_series=fm)
    return fm[fm > 0.99*epsilon].index.to_list()


# def non_convex_compute_obj(v: np.array, rho: np.array, epsilon: float):
#     return np.minimum(abs(v)/epsilon, np.ones(v.shape)).dot(rho)


def non_convex_LP7(J,
                   model: cobra.Model,
                   epsilon: float,
                   use_abs=True,
                   flux_logger=None) -> list:
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
    if flux_logger is not None:
        flux_logger.add(name="nc_LP7", flux_series=fm)
    return fm[fm > 0.99*epsilon].index.to_list()


def LP7(J,
        model: cobra.Model,
        epsilon: float,
        use_abs=True,
        rxn_scale_eps=None,
        tol_coef=0.99,
        return_min_v=False,
        flux_logger=None) -> list:
    """
    LP7 tries to maximize the number of feasible fluxes in J whose value is at least epsilon (Nikos Vlassis, et al. 2013)

    Parameters
    ----------
    J
        A rxn set that the number of feasible fluxes in it is maximized by LP7
    model
        Used cobra model that contains all the J
    epsilon
        Threshold of non-zero fluxes
    use_abs
        If True, the returned reactions are selected by its absolute value
    tol_coef


    Returns
    -------
        a list of rxn IDs, the rxns can produce feasible fluxes.
    """

    with model:
        prob = model.problem
        vars = []
        consts = []
        constr_coefs = {}
        for j in J:
            rxn = model.reactions.get_by_id(j)

            # we make the lb negative because some reactions need to be negative to produce valid fluxes sometimes

            if isinstance(epsilon, pd.Series):
                var = prob.Variable(f"LP7_z_{rxn.id}", lb=-np.inf, ub=epsilon[j])
            else:
                var = prob.Variable(f"LP7_z_{rxn.id}", lb=-np.inf, ub=epsilon)  # -inf <= zi <= eps

            c_name = f"const_LP7_{rxn.id}"
            const = prob.Constraint(Zero,
                                    name=c_name, ub=0)  # vi >= zi
            consts.append(const)
            constr_coefs[c_name] = {var: 1,
                                    rxn.forward_variable: -1,
                                    rxn.reverse_variable: 1}
            vars.append(var)
        model.add_cons_vars(consts, sloppy=True)
        model.add_cons_vars(vars, sloppy=True)
        for con, coefs in constr_coefs.items():
            model.constraints[con].set_linear_coefficients(coefs)

        model.objective = prob.Objective(Zero, sloppy=True)
        model.objective.set_linear_coefficients({v: 1 for v in vars})
        model.solver.update()
        sol = model.optimize(objective_sense="maximize", raise_error=True)
    if use_abs:
        fm = sol.to_frame()["fluxes"].abs().sort_index()
    else:
        fm = sol.to_frame()["fluxes"].sort_index()

    if isinstance(epsilon, pd.Series):
        epsilon = epsilon[fm.index]

    if flux_logger is not None:
        rxn_name = list(J)[0]
        flux_logger.add(name=f"LP7_{rxn_name}", flux_series=fm)
    if rxn_scale_eps is None:
        if return_min_v:
            return fm[fm > tol_coef*epsilon].index.to_list(), fm
        return fm[fm > tol_coef*epsilon].index.to_list()
    return fm[fm > rxn_scale_eps[fm.index]].index.to_list()


def LP9(K: np.ndarray,
        P: np.ndarray,
        NonP: np.ndarray,
        model: cobra.Model,
        epsilon,
        min_v_ser,
        scaling_factor = 1,
        tol_coef=0.99,
        rxn_scale_eps=None,
        flux_logger=None) -> list:
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
    if isinstance(epsilon, pd.Series):
        epsilon = epsilon.sort_index()
    assert (min_v_ser > 0).all()
    assert len(np.intersect1d(P, K)) == 0
    with model:
        prob = model.problem
        vars = []
        objs = []
        constr_coefs = {}
        # print(f"In LP9, |P| = {len(P)}, |K| = {len(K)}")

        for p in P:  # penalized rxns
            rxn = model.reactions.get_by_id(p)
            f_name, r_name = f"const_forward_LP9_{rxn.id}", f"const_reverse_LP9_{rxn.id}"
            fconst = prob.Constraint(Zero,
                                     name=f_name, ub=0)  # vi <= zi
            rconst = prob.Constraint(Zero,
                                     name=r_name, ub=0)  # -zi <= vi
            model.add_cons_vars([fconst, rconst])
            z = prob.Variable(f"LP9_z_{rxn.id}", lb=0, ub=max(abs(rxn.lower_bound),
                                                              abs(rxn.upper_bound)) * scaling_factor)  # define z
            constr_coefs[fconst] = {z: -1.0,
                                    rxn.forward_variable: 1.0,
                                    rxn.reverse_variable: -1.0}
            constr_coefs[rconst] = {z: -1.0,
                                    rxn.forward_variable: -1.0,
                                    rxn.reverse_variable: 1.0}
            vars.append(z)
            if p not in NonP:
                objs.append(z)
        model.add_cons_vars(vars)
        model.solver.update()
        for r in model.reactions:
            if r.id in K:
                r.upper_bound = max(r.upper_bound, min_v_ser[r.id]) * scaling_factor
                r.lower_bound = min_v_ser[r.id] * scaling_factor
            else:
                r.lower_bound *= scaling_factor
            r.upper_bound *= scaling_factor
        for con, coefs in constr_coefs.items():
            con.set_linear_coefficients(coefs)
        model.objective = prob.Objective(Zero, sloppy=True)
        model.objective.set_linear_coefficients({v: -1.0 for v in objs})  # sum of zi

        try:
            sol = model.optimize(objective_sense="maximize", raise_error=True)
        except:
            print("infeasible result: |K| = ", len(K))
            return []
        fm = sol.to_frame()["fluxes"].abs().sort_index()
    if flux_logger is not None:
        flux_logger.add(name="LP9", flux_series=fm)

    if isinstance(epsilon, pd.Series):
        epsilon = epsilon[fm.index]
    if rxn_scale_eps is None:
        return fm[fm > tol_coef * np.minimum(epsilon, min_v_ser.min())].index.to_list()  # supp
    return fm[fm > rxn_scale_eps[fm.index]].index.to_list()


def find_sparse_mode(J, P, nonP, model, singleJ, epsilon):
    if len(J) == 0:
        return []
        # print(f"find_sparse_mode of single reaction: {singleJ}")
    supps, v = LP7(J if singleJ is None else singleJ,
                   model, epsilon, use_abs=False, return_min_v=True)
    K = np.intersect1d(J if singleJ is None else singleJ, supps)  # J might not be an irrv set
    if singleJ is not None and len(np.intersect1d(singleJ, K)) == 0:
        warn(f"Singleton {singleJ} flux cannot be generated in LP7")
    if K.shape[0] == 0:
        return []
    return LP9(K, P, nonP, model, epsilon, min_v_ser=v[list(K)])