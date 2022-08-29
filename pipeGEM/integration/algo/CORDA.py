import copy
from tqdm import tqdm
from typing import Dict, List

import pandas as pd
import numpy as np
from scipy import interpolate
import cobra
from optlang.symbolics import Zero

from pipeGEM.analysis import timing, CORDA_Analysis
from pipeGEM.integration.utils import parse_predefined_threshold


class CORDABuilder:
    def __init__(self,
                 model,
                 conf_scores,
                 mocks=None,
                 n_iters=np.inf,
                 penalty_factor=100,
                 penalty_increase_factor=1.1,
                 upper_bound=1e6,
                 support_flux_value=1,
                 threshold=1e-6):
        self._model = model
        self._conf_scores = conf_scores
        self._n = n_iters
        self._n_redundancies = {}
        self._pf = penalty_factor
        self._pif = penalty_increase_factor
        self._upper_bound = upper_bound
        self._sf = support_flux_value
        self._infeasible_rxns = []
        self._threshold = threshold
        self._mocks = mocks if mocks is not None else []
        self._bounds = {r.id: r.bounds for r in model.reactions}
        for r in model.reactions:
            r.upper_bound = self._upper_bound if r.upper_bound > threshold else r.upper_bound
            r.lower_bound = -self._upper_bound if -r.lower_bound > threshold else r.lower_bound

    @property
    def model(self):
        return self._model

    @property
    def conf_scores(self):
        return self._conf_scores

    def _get_support_rxns(self,
                          var_ids,
                          keep_if_support=None,
                          penalize_medium_score=True,
                          support_redundancies=True):
        penalty_dic = {var_id: (1 if penalize_medium_score else 0)
                       if self._conf_scores[var_id] >= 0 else self._pf
                       for var_id in var_ids if self._conf_scores[var_id] <= 2}

        n_iters = self._n if support_redundancies else 1
        all_support_vars = set()

        if keep_if_support is not None:
            eval_vars = [i for i, c in self._conf_scores.items() if c < 0]
            n_sups = pd.DataFrame({"n": np.zeros(shape=(len(eval_vars),))}, index=eval_vars)

        for var_id in var_ids:
            self._model.objective = Zero
            var = self._model.variables[var_id]
            if var.ub < self._threshold:
                self._infeasible_rxns.append(var_id)
                self._conf_scores[var_id] = -1
                continue
            old_bounds = (var.lb, var.ub)
            var.lb = max(self._sf, var.lb)
            var.ub = self._upper_bound
            cur_penalty = {self._model.variables[k]: v for k, v in penalty_dic.items()}
            all_support_vars_for_v = set()
            for i in range(n_iters):
                self._model.objective.set_linear_coefficients(cur_penalty)
                sol = self._model.solver.optimize()
                if sol != "optimal":
                    self._infeasible_rxns.append(var_id)
                    self._conf_scores[var_id] = -1
                    break
                sol = self._model.solver.primal_values
                support_vars = set([v for v in sol if sol[v] > self._threshold
                                    and self._conf_scores[v] <= 2 and v != var_id])
                new_sups = support_vars - all_support_vars_for_v
                all_support_vars_for_v |= support_vars
                if keep_if_support is not None:
                    n_sups.loc[set(eval_vars) & support_vars, "n"] += 1

                if len(new_sups) == 0:
                    self._n_redundancies[var_id] = i+1 if support_redundancies else self._n_redundancies[var_id]
                    break
                for v in new_sups:
                    if self._model.variables[v] in cur_penalty:
                        cur_penalty[self._model.variables[v]] = cur_penalty[self._model.variables[v]] * self._pif
            all_support_vars |= all_support_vars_for_v
            var.lb, var.ub = old_bounds
        if keep_if_support is None:
            return all_support_vars
        return all_support_vars, set(n_sups.query(f"n >= {keep_if_support}").index)

    def build(self,
              keep_if_support=5):
        hi_supports = self._get_support_rxns([i for i, c in self._conf_scores.items() if c >= 3])
        for i in hi_supports:
            self._conf_scores[i] = 3

        m_l_supports, to_keeps = self._get_support_rxns([i
                                                         for i, c in self._conf_scores.items() if 0 <= c < 3],
                                                        keep_if_support=keep_if_support,
                                                        penalize_medium_score=False)
        for i in to_keeps:
            self._conf_scores[i] = 3
        low_confs = [i for i, c in self._conf_scores.items() if i < 0]
        for vid in low_confs:
            v = self._model.variables[vid]
            v.ub = max(0.0, v.lb)

        self._model.objective = Zero

        #TODO
        for v in self._model.variables:
            if 0 < self._conf_scores[v.name] < 3:
                self._model.objective.set_linear_coefficients({v: 1})
                sol = self._model.solver.optimize()
                if sol == "optimal" and self._model.objective.value > self._sf:
                    self._conf_scores[v.name] = 3
            self._model.objective.set_linear_coefficients({v: 0})

        for vid, conf in self._conf_scores.items():
            if 0 < conf < 3:
                self._model.variables[vid].ub = 0.0
            elif conf == 0:
                self._conf_scores[vid] = -1
        hi_supports = self._get_support_rxns([i for i, c in self._conf_scores.items() if c >= 3],
                                             penalize_medium_score=False,
                                             support_redundancies=False)
        for i in hi_supports:
            self._conf_scores[i] = 3

        to_remove = []
        for rxn in self._model.reactions:
            conf = max(self._conf_scores[rxn.id], self._conf_scores[rxn.reverse_id])
            if conf == 3 and rxn not in self._mocks:
                rxn.bounds = self._bounds[rxn.id]
            else:
                to_remove.append(rxn)
        self._model.remove_reactions(to_remove, remove_orphans=True)
        return self._model


def _add_prod_met_rxns(model,
                       met_prod,
                       conf_scores,
                       upper_bound):
    mocks = []
    conf_scores = {k: v for k, v in conf_scores.items()}
    for mid in met_prod:
        r = cobra.Reaction("EX_CORDA_" + mid)
        r.notes["mock"] = mid
        r.upper_bound = upper_bound
        model.add_reactions([r])
        mocks.append(r)
        r.add_metabolites({model.metabolites.get_by_id(mid): -1})
        conf_scores[r.id] = 3
    return mocks, conf_scores


class DiscreteStrategy:
    def __init__(self,
                 exp_thres,
                 nonexp_thres,
                 rxn_scores):
        self._exp = exp_thres
        self._nexp = nonexp_thres
        self._rxn_scores = rxn_scores

    def transform(self):
        raise NotImplementedError()


class LinearDiscreteStrategy(DiscreteStrategy):
    def __init__(self,
                 exp_thres,
                 nonexp_thres,
                 rxn_scores
                 ):
        super().__init__(exp_thres,
                         nonexp_thres,
                         rxn_scores)

    def transform(self):
        f = interpolate.interp1d([self._exp, self._nexp], [3, -1], fill_value='extrapolate')
        return {r: f(v) if np.isfinite(v) else 0 for r, v in self._rxn_scores.items()}


discrete_strategies = {"linear": LinearDiscreteStrategy}

@timing
def apply_CORDA(model,
                data,
                predefined_threshold = None,
                use_heuristic_th = False,
                discrete_strategy_name: str = "linear",
                protected_rxns=None,
                n_iters=np.inf,
                penalty_factor=100,
                penalty_increase_factor=1.1,
                keep_if_support=5,
                met_prod = None,
                upper_bound=1e6,
                threshold=1e-6,
                support_flux_value=1,
                ) -> CORDA_Analysis:
    mocks = []
    threshold_dic = parse_predefined_threshold(predefined_threshold,
                                               gene_data=data.gene_data,
                                               use_heuristic_th=use_heuristic_th)
    th_result, exp_th, non_exp_th = threshold_dic["th_result"], threshold_dic["exp_th"], threshold_dic["non_exp_th"]
    transformer = discrete_strategies[discrete_strategy_name](exp_th, non_exp_th, data.rxn_scores)
    conf_scores = transformer.transform()
    model = model.copy()
    # check
    if met_prod is not None:
        mocks, conf_scores = _add_prod_met_rxns(model,
                                                met_prod,
                                                conf_scores,
                                                upper_bound)
    for r in model.reactions:
        if r.objective_coefficient != 0:
            conf_scores[r.id] = 3

    if protected_rxns is not None:
        for r in protected_rxns:
            conf_scores[r] = 3

    var_conf_scores = {r: conf for r, conf in conf_scores.items()}
    var_conf_scores.update({model.reactions.get_by_id(r).reverse_id: conf for r, conf in conf_scores.items()})

    # init corda builder and build the model
    corda_builder = CORDABuilder(model,
                                 conf_scores=var_conf_scores,
                                 mocks=mocks,
                                 n_iters=n_iters,
                                 penalty_factor=penalty_factor,
                                 penalty_increase_factor=penalty_increase_factor,
                                 upper_bound=upper_bound,
                                 support_flux_value=support_flux_value,
                                 threshold=threshold
                                 )
    result_model = corda_builder.build(keep_if_support=keep_if_support)
    conf_scores = corda_builder.conf_scores
    result = CORDA_Analysis(log={"n_iters": n_iters,
                                 "penalty_factor": penalty_factor,
                                 "penalty_increase_factor": penalty_increase_factor,
                                 "keep_if_support": keep_if_support,
                                 "met_prod": met_prod,
                                 "upper_bound": upper_bound,
                                 "threshold": threshold,
                                 "support_flux_value": support_flux_value})
    result.add_result(model=result_model, conf_scores=conf_scores)
    return result