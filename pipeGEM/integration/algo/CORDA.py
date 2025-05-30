import copy
import warnings

from tqdm import tqdm
from typing import Dict, List

import pandas as pd
import numpy as np
from scipy import interpolate
import cobra
from optlang.symbolics import Zero

from pipeGEM.analysis import timing, CORDA_Analysis, measure_efficacy
from pipeGEM.integration.utils import parse_predefined_threshold


class CORDABuilder:
    """Builds a context-specific model using the CORDA algorithm.

    Implements the core logic of CORDA (Cost Optimization Reaction
    Dependency Assessment). Iteratively identifies reactions required to
    support high-confidence reactions and removes reactions associated
    with low-confidence data.

    Parameters
    ----------
    model : cobra.Model
        Input genome-scale metabolic model.
    conf_scores : dict
        Mapping of reaction variable IDs (forward and reverse) to confidence scores.
        Expected scores: -1 (low), 0 (medium-low), 1 (medium-high), 3 (high/core).
    mocks : list, optional
        List of mock reaction objects (e.g., for metabolite production).
        Defaults to None.
    n_iters : int, optional
        Maximum iterations for finding support reactions per core reaction.
        Defaults to infinity (effectively number of reactions).
    penalty_factor : float, optional
        Initial penalty for low-confidence reactions in objective. Defaults to 100.
    penalty_increase_factor : float, optional
        Factor to increase penalty each iteration. Defaults to 1.1.
    upper_bound : float, optional
        High upper bound for reactions during optimization. Defaults to 1e6.
    support_flux_value : float or dict, optional
        Minimum flux for a reaction to be considered 'supporting'.
        Can be float or dict mapping reaction IDs to values if `rxn_scaling_coefs`
        is provided. Defaults to 1.
    threshold : float, optional
        Flux threshold below which flux is considered zero. Defaults to 1e-6.
    skip_last_step : bool, optional
        If True, skip final refinement step removing medium-confidence reactions
        not supporting the final core set. Defaults to True.
    rxn_scaling_coefs : dict, optional
        Mapping of reaction IDs to scaling coefficients to adjust
        `support_flux_value` per reaction. Defaults to None.

    Attributes
    ----------
    model : cobra.Model
        The cobra.Model object being processed.
    conf_scores : dict
        Dictionary of confidence scores for reaction variables.

    """
    def __init__(self,
                 model,
                 conf_scores,
                 mocks=None,
                 n_iters=np.inf,
                 penalty_factor=100,
                 penalty_increase_factor=1.1,
                 upper_bound=1e6,
                 support_flux_value=1,
                 threshold=1e-6,
                 skip_last_step=True,
                  rxn_scaling_coefs=None):
        self._model = model
        self._conf_scores = conf_scores
        self._n = n_iters if np.isfinite(n_iters) else len(model.reactions)
        self._n_redundancies = {}
        self._pf = penalty_factor
        self._pif = penalty_increase_factor
        self._upper_bound = upper_bound
        self._sf = support_flux_value if rxn_scaling_coefs is None else \
            {k: support_flux_value / v for k, v in rxn_scaling_coefs.items()}
        self._infeasible_rxns = []
        self._threshold = threshold
        self._mocks = mocks if mocks is not None else []
        self._bounds = {r.id: r.bounds for r in model.reactions}
        for r in model.reactions:
            r.upper_bound = self._upper_bound if r.upper_bound > threshold else r.upper_bound
            r.lower_bound = -self._upper_bound if -r.lower_bound > threshold else r.lower_bound
        self._skip_last_step = skip_last_step

    # Properties docstrings are now part of the class docstring Attributes section
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
                       for var_id in var_ids if self._conf_scores[var_id] < 3}

        n_iters = self._n if support_redundancies else 1
        all_support_vars = set()

        if keep_if_support is not None:
            eval_vars = [i for i, c in self._conf_scores.items() if c < 0]
            n_sups = pd.DataFrame({"n": np.zeros(shape=(len(eval_vars),))}, index=eval_vars)

        for vi, var_id in tqdm(enumerate(var_ids), desc="Variables to check"):
            if n_iters == 1 and var_id in all_support_vars:
                continue

            self._model.objective = Zero
            var = self._model.variables[var_id]
            if var.ub < self._threshold:
                self._infeasible_rxns.append(var_id)
                self._conf_scores[var_id] = -1
                continue
            old_bounds = (var.lb, var.ub)
            var.ub = self._upper_bound
            min_f = self._sf[var_id] if isinstance(self._sf, dict) else self._sf
            var.lb = max(min_f, var.lb)
            cur_penalty = {self._model.variables[k]: v for k, v in penalty_dic.items()}
            all_support_vars_for_v = set()
            for i in range(n_iters):
                self._model.objective.set_linear_coefficients(cur_penalty)
                self._model.objective.direction = "min"
                sol = self._model.solver.optimize()
                if sol != "optimal":
                    self._infeasible_rxns.append(var_id)
                    self._conf_scores[var_id] = -1
                    break
                sol = self._model.solver.primal_values
                support_vars = set([v for v in sol if abs(sol[v]) >= self._threshold
                                    and self._conf_scores[v] < 3 and v != var_id])
                new_sups = support_vars - all_support_vars_for_v
                all_support_vars_for_v |= support_vars
                if keep_if_support is not None:
                    n_sups.loc[list(set(eval_vars) & support_vars), "n"] += 1

                if len(new_sups) == 0:
                    if support_redundancies:
                        self._n_redundancies[var_id] = i+1
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
        """Execute the CORDA algorithm steps to build the context-specific model.

        Performs the main steps of CORDA:
        1. Identify reactions supporting high-confidence (core) reactions.
        2. Identify reactions supporting medium-confidence reactions and elevate
           those supported by many core reactions (`keep_if_support`).
        3. (Optional) Refine model by removing medium-confidence reactions not
           supporting the final core set.
        4. Remove low-confidence reactions and reactions not part of the final
           core/support set.

        Parameters
        ----------
        keep_if_support : int, optional
            Minimum number of high-confidence reactions a medium-confidence
            reaction must support to be elevated to high confidence. Defaults to 5.

        Returns
        -------
        tuple
            - cobra.Model: Resulting context-specific model.
            - numpy.ndarray: Array of reaction objects removed from the original model.
        """
        n_cores = len([i for i, c in self._conf_scores.items() if c >= 3])
        hi_supports = self._get_support_rxns([i for i, c in self._conf_scores.items() if c >= 3])
        print(f"{n_cores - len([i for i, c in self._conf_scores.items() if c >= 3])}"
              f" were removed from the core variables")
        n_cores = len([i for i, c in self._conf_scores.items() if c >= 3])
        for i in hi_supports:
            self._conf_scores[i] = 3
        print(f"step 1 finished, {len([i for i, c in self._conf_scores.items() if c >= 3]) - n_cores}"
              f"non-core variables were added to the core variables")
        n_cores = len([i for i, c in self._conf_scores.items() if c >= 3])
        m_l_supports, to_keeps = self._get_support_rxns([i
                                                         for i, c in self._conf_scores.items() if 0 <= c < 3],
                                                        keep_if_support=keep_if_support,
                                                        penalize_medium_score=False)
        for i in to_keeps:
            self._conf_scores[i] = 3
        low_confs = [i for i, c in self._conf_scores.items() if c < 0]
        for vid in low_confs:
            v = self._model.variables[vid]
            v.ub = max(0.0, v.lb)

        self._model.objective = Zero

        for v in self._model.variables:
            if 0 < self._conf_scores[v.name] < 3:
                self._model.objective.set_linear_coefficients({v: 1})
                sol = self._model.solver.optimize()
                min_f = self._sf[v.name] if isinstance(self._sf, dict) else self._sf
                if sol == "optimal" and self._model.objective.value > min_f:
                    sol = self._model.solver.primal_values
                    if isinstance(self._sf, dict):
                        to_change = [vi for vi in sol if sol[vi] > self._sf[vi] and 0 < self._conf_scores[vi] < 3]
                    else:
                        to_change = [vi for vi in sol if sol[vi] > min_f and 0 < self._conf_scores[vi] < 3]
                    for vi in to_change:
                        self._conf_scores[vi] = 3
            self._model.objective.set_linear_coefficients({v: 0})
        print(f"step 2 finished, {len([i for i, c in self._conf_scores.items() if c >= 3]) - n_cores}"
              f"non-core variables were added to core variables")
        n_cores = len([i for i, c in self._conf_scores.items() if c >= 3])
        # do we need this?
        if not self._skip_last_step:
            for vid, conf in self._conf_scores.items():
                if 0 < conf < 3:
                    self._model.variables[vid].ub = 0.0
                elif conf == 0:
                    self._conf_scores[vid] = -1
            hi_supports = self._get_support_rxns([i for i, c in self._conf_scores.items() if c >= 3],
                                                 penalize_medium_score=False,
                                                 support_redundancies=False)
            print(f"step 3 finished, {len([i for i, c in self._conf_scores.items() if c >= 3]) - n_cores} "
                  f"non-core variables were added to core variables")
            for i in hi_supports:
                self._conf_scores[i] = 3

        to_remove = []
        for rxn in self._model.reactions:
            conf = max(self._conf_scores[rxn.id], self._conf_scores[rxn.reverse_id])
            if conf >= 3 and rxn not in self._mocks:
                rxn.bounds = self._bounds[rxn.id]
            else:
                to_remove.append(rxn)
        self._model.remove_reactions(to_remove, remove_orphans=True)
        return self._model, np.array(to_remove)


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
    """Base class for strategies converting continuous scores to discrete levels.

    Parameters
    ----------
    exp_thres : float
        Threshold above which score indicates high confidence (expressed).
    nonexp_thres : float
        Threshold below which score indicates low confidence (not expressed).
    rxn_scores : dict
        Mapping of reaction IDs to continuous scores (e.g., expression values).
    """
    def __init__(self,
                 exp_thres,
                 nonexp_thres,
                 rxn_scores):
        self._exp = exp_thres
        self._nexp = nonexp_thres
        self._rxn_scores = rxn_scores

    def transform(self):
        """Transform continuous reaction scores into discrete confidence levels.

        Must be implemented by subclasses.

        Returns
        -------
        dict
            Mapping of reaction IDs to discrete confidence scores (-1, 0, 1, 3).

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError()


class LinearDiscreteStrategy(DiscreteStrategy):
    """Convert continuous scores to discrete levels using linear interpolation.

    Parameters
    ----------
    exp_thres : float
        Threshold above which score indicates high confidence (expressed).
    nonexp_thres : float
        Threshold below which score indicates low confidence (not expressed).
    rxn_scores : dict
        Mapping of reaction IDs to continuous scores.
    """
    def __init__(self,
                 exp_thres,
                 nonexp_thres,
                 rxn_scores
                 ):
        super().__init__(exp_thres,
                         nonexp_thres,
                         rxn_scores)

    def transform(self):
        """Transform continuous scores using linear interpolation between thresholds.

        Scores above `exp_thres` map to 3.
        Scores below `nonexp_thres` map to -1.
        Scores between thresholds are linearly interpolated between -1 and 3.
        Infinite scores map to 0.

        Returns
        -------
        dict
            Mapping of reaction IDs to discrete confidence scores (-1, 0, 1, 3).
        """
        f = interpolate.interp1d([self._exp, self._nexp], [3, -1], fill_value='extrapolate')
        res = {}
        for r, v in self._rxn_scores.items():
            if np.isfinite(v):
                res[r] = f(v)
                if isinstance(res[r], np.ndarray):
                    res[r] = res[r][()]
            else:
                res[r] = 0
        return res


discrete_strategies = {"linear": LinearDiscreteStrategy}


@timing
def apply_CORDA(model,
                data,
                protected_rxns=None,
                predefined_threshold = None,
                threshold_kws = None,
                rxn_scaling_coefs=None,
                discrete_strategy_name: str = "linear",
                n_iters=np.inf,
                penalty_factor=100,
                penalty_increase_factor=1.1,
                keep_if_support=5,
                met_prod = None,
                upper_bound=1e6,
                threshold=1e-6,
                support_flux_value=1,
                skip_last_step=True,
                ) -> CORDA_Analysis:
    """Apply the CORDA algorithm to generate a context-specific metabolic model.

    Orchestrates the CORDA process:
    1. Prepare model and data (thresholds, confidence scores, protected reactions).
    2. Initialize and run CORDABuilder.
    3. Calculate efficacy metrics.
    4. Return results in a CORDA_Analysis object.

    Parameters
    ----------
    model : cobra.Model
        Input genome-scale metabolic model.
    data : object
        Object with reaction scores (`data.rxn_scores`) and optionally
        gene data (`data.gene_data`) if using gene-based thresholds.
    protected_rxns : list[str], optional
        Reaction IDs to force into the core set (confidence 3). Defaults to None.
    predefined_threshold : str, optional
        Name of predefined thresholding strategy (e.g., 'percentile_90').
        Requires `data.gene_data`. Defaults to None.
    threshold_kws : dict, optional
        Additional keyword arguments for the thresholding function
        (used with `predefined_threshold`). Defaults to None.
    rxn_scaling_coefs : dict, optional
        Mapping of reaction IDs to scaling coefficients to adjust flux thresholds.
        Defaults to None.
    discrete_strategy_name : str, optional
        Strategy to convert continuous scores to discrete confidence levels
        ('linear'). Defaults to "linear".
    n_iters : int, optional
        Max iterations for finding support reactions in CORDABuilder.
        Defaults to infinity.
    penalty_factor : float, optional
        Initial penalty factor in CORDABuilder. Defaults to 100.
    penalty_increase_factor : float, optional
        Penalty increase factor in CORDABuilder. Defaults to 1.1.
    keep_if_support : int, optional
        Support threshold for elevating medium confidence in CORDABuilder.
        Defaults to 5.
    met_prod : list[str], optional
        Metabolite IDs for which mock production reactions should be added
        and forced into the core set. Defaults to None.
    upper_bound : float, optional
        High upper bound for reactions during optimization. Defaults to 1e6.
    threshold : float, optional
        Flux threshold below which flux is considered zero. Defaults to 1e-6.
    support_flux_value : float or dict, optional
        Minimum flux required for support reactions in CORDABuilder.
        Defaults to 1.
    skip_last_step : bool, optional
        Whether to skip the final refinement step in CORDABuilder.
        Defaults to True.

    Returns
    -------
    CORDA_Analysis
        Object containing results: context-specific model, confidence scores,
        removed reactions, efficacy metrics, and logs.

    Note
    -------
    Original paper: Schultz, A., & Qutub, A. A. (2016).
    Reconstruction of tissue-specific metabolic networks using CORDA.
    PLoS computational biology, 12(3), e1004808.
    """
    mocks = []
    threshold_kws = threshold_kws or {}
    threshold_dic = parse_predefined_threshold(predefined_threshold,
                                               gene_data=data.gene_data,
                                               **threshold_kws)
    th_result, exp_th, non_exp_th = threshold_dic["th_result"], threshold_dic["exp_th"], threshold_dic["non_exp_th"]
    transformer = discrete_strategies[discrete_strategy_name](exp_th, non_exp_th, data.rxn_scores)
    conf_scores = transformer.transform()
    model = model.copy()
    old_exchange_bounds = {r.id: r.bounds for r in model.exchanges}
    # check
    if met_prod is not None:
        mocks, conf_scores = _add_prod_met_rxns(model,
                                                met_prod,
                                                conf_scores,
                                                upper_bound)
    obj_coefs = {}
    for r in model.reactions:
        if r.objective_coefficient != 0:
            conf_scores[r.id] = 3
            obj_coefs[r.forward_variable] = r.objective_coefficient
            obj_coefs[r.reverse_variable] = -r.objective_coefficient

    if protected_rxns is not None:
        rxn_id_in_model = [r.id for r in model.reactions]
        for r in protected_rxns:
            if r in rxn_id_in_model:
                conf_scores[r] = 3
            else:
                warnings.warn(f"{r} is not in the model")
    core_ids = [r for r, conf in conf_scores.items() if conf >= 3 ]
    ncore_ids = [r for r, conf in conf_scores.items() if conf < 0]
    var_conf_scores = {r: conf for r, conf in conf_scores.items()}
    var_conf_scores.update({model.reactions.get_by_id(r).reverse_id: conf for r, conf in conf_scores.items()})
    if rxn_scaling_coefs is not None:
        var_rxn_scaling_coefs = {model.reactions.get_by_id(r).id: coef
                                 for r, coef in rxn_scaling_coefs.items()}
        var_rxn_scaling_coefs.update({model.reactions.get_by_id(r).reverse_id: coef
                                      for r, coef in rxn_scaling_coefs.items()})
    else:
        var_rxn_scaling_coefs = None

    # init corda builder and build the model
    corda_builder = CORDABuilder(model,
                                 conf_scores=var_conf_scores,
                                 mocks=mocks,
                                 n_iters=n_iters,
                                 penalty_factor=penalty_factor,
                                 penalty_increase_factor=penalty_increase_factor,
                                 upper_bound=upper_bound,
                                 support_flux_value=support_flux_value,
                                 threshold=threshold,
                                 skip_last_step=skip_last_step,
                                 rxn_scaling_coefs=var_rxn_scaling_coefs
                                 )
    result_model, removed_rxn_ids = corda_builder.build(keep_if_support=keep_if_support)
    result_model.objective.set_linear_coefficients(obj_coefs)
    result_model.objective.direction = "max"
    for r in result_model.exchanges:
        if r.id in old_exchange_bounds:
            result_model.reactions.get_by_id(r.id).bounds = old_exchange_bounds[r.id]

    conf_scores = corda_builder.conf_scores
    result = CORDA_Analysis(log={"n_iters": n_iters,
                                 "penalty_factor": penalty_factor,
                                 "penalty_increase_factor": penalty_increase_factor,
                                 "keep_if_support": keep_if_support,
                                 "met_prod": met_prod,
                                 "upper_bound": upper_bound,
                                 "flux_threshold": threshold,
                                 "support_flux_value": support_flux_value})
    algo_efficacy = measure_efficacy(kept_rxn_ids=[r.id for r in result_model.reactions],
                                     removed_rxn_ids=removed_rxn_ids,
                                     core_rxn_ids=core_ids,
                                     non_core_rxn_ids=ncore_ids)
    result.add_result(dict(result_model=result_model,
                           conf_scores=conf_scores,
                           threshold_analysis=th_result,
                           removed_rxn_ids=removed_rxn_ids,
                           algo_efficacy=algo_efficacy))
    return result
