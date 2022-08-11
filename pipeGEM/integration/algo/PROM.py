import warnings
from typing import Union, Sequence, Dict
from dataclasses import dataclass

from scipy import stats
import cobra
from cobra.flux_analysis import flux_variability_analysis

import pandas as pd
import numpy as np

from pipeGEM.core import Problem
from pipeGEM.integration.utils import get_PROM_threshold


def quantile_norm(df: pd.DataFrame) -> pd.DataFrame:
    rank_mean = df.stack().groupby(df.rank(method='first').stack().astype(int)).mean()
    return df.rank(method='min').stack().astype(int).map(rank_mean).unstack()


def _check_regulators(data_df, regulation_df) -> dict:
    problematic_regulators = np.setdiff1d(regulation_df.iloc[:, 0].values, data_df.index.values)
    problematic_targets = np.setdiff1d(regulation_df.iloc[:, 1].values, data_df.index.values)
    if len(problematic_regulators) != 0 or len(problematic_targets) != 0:
        warnings.warn(f"some of the regulators or targets are not in the data index\n"
                      f"regulators: {problematic_regulators}\n"
                      f"targets: {problematic_targets}")
    return {"regulators": problematic_regulators, "targets": problematic_targets}


def _check_error_rate(has_errs):
    if np.sum(has_errs) / len(has_errs) > 0.75:
        raise ValueError("Error rate > 0.75, please change the binarization threshold")


def get_gene_prob(data: pd.DataFrame,  # shape = (G, S)
                  regulation: pd.DataFrame,  # shape = (TG, 2)
                  regulator_subsets: list,
                  ks_threshold: float = 0.05,
                  q: float = 0.33) -> Dict[str, pd.DataFrame]:
    # return reg pairs prob list
    # regulators checking
    regulation = regulation[regulation.iloc[:, 0].isin(regulator_subsets) &
                            regulation.iloc[:, 1].isin(regulator_subsets)]
    problematic_dic = _check_regulators(data, regulation)
    data = quantile_norm(data.copy())
    threshold = get_PROM_threshold(data, q)
    binary_data = pd.DataFrame(np.where(data >= threshold, 1, 0),
                               columns=data.columns,
                               index=data.index)
    has_errs = np.zeros((regulation.shape[0], ))
    prob_tf_genes = np.ones((regulation.shape[0], ))
    for i, (regulator, target) in regulation.iterrows():
        if regulator in problematic_dic["regulators"] or target in problematic_dic["targets"]:
            has_errs[i] = 1
            continue

        target_data = data.loc[target, :]
        # regulator_data = data.loc[regulator, :]
        target_binary = binary_data[target, :]
        regulator_binary = binary_data[regulator, :]
        ks2_result = stats.ks_2samp(target_data[regulator_binary == 1], target_data[regulator_binary == 0])
        if ks2_result.pvalue < ks_threshold:
            prob_tf_genes[i] = sum(target_binary[regulator_binary == 1]) / len(target_binary)
    return {"prob_df": pd.concat([regulation,
                                  pd.DataFrame({"probability": prob_tf_genes,
                                                "has_err": has_errs})], axis=1),
            }


def _get_long_lb_ub(n_rxns: int) -> (np.ndarray, np.ndarray):
    return np.concatenate([-1000 * np.ones(n_rxns), np.zeros(n_rxns * 2)]), \
           np.concatenate([1000 * np.ones(n_rxns), np.zeros(n_rxns * 2)])


@dataclass
class PROMProb(Problem):
    model: cobra.Model
    kappa: float = 1
    threshold: float = 1e-6
    m_threshold: float = 1e-3

    def __post_init__(self):
        self.ori_lbs = np.array([r.lower_bound for r in self.model.reactions])
        self.ori_ubs = np.array([r.upper_bound for r in self.model.reactions])
        self.gene_ids = [g.id for g in self.model.genes]
        self.gene_nums = {g_id: i for i, g_id in enumerate(self.gene_ids)}
        self.rxn_ids = [r.id for r in self.model.reactions]
        self.rxn_nums = {r_id: i for i, r_id in enumerate(self.rxn_ids)}
        self.prepare_problem()

    def prepare_problem(self):
        """
        A          lb     f    ub      b
        S  Z  Z    -1000  v    1000    =0
        E  E  Z    0      va   0       >lb
        E  Z -E    0      vb   0       <ub

        :return:
        """

        super().prepare_problem()
        self.ori_lbs = np.where(self.ori_lbs == self.ori_ubs, self.ori_ubs - self.threshold, self.ori_lbs)
        a1 = np.concatenate([self.S,
                             np.zeros(self.S.shape),
                             np.zeros(self.S.shape)], axis=1)
        a2 = np.concatenate([np.eye(self.S.shape[1]),
                             np.eye(self.S.shape[1]),
                             np.zeros(self.S.shape[1], self.S.shape[1])], axis=1)
        a3 = np.concatenate([np.eye(self.S.shape[1]),
                             np.zeros(self.S.shape[1],
                                      self.S.shape[1]),
                             -np.eye(self.S.shape[1])], axis=1)
        self.A = np.concatenate([a1, a2, a3])
        self.long_objs = np.array([r.objective_coefficient for r in self.model.reactions] +
                                 [0 for _ in range(len(self.model.reactions) * 2)])
        self.long_b = np.concatenate([np.zeros(len(self.S.shape[0])), self.ori_lbs, self.ori_ubs])
        self.long_csense = ["E" for _ in range(len(self.S.shape[0]))] + \
                           ["G" for _ in range(len(self.model.reactions))] + \
                           ["L" for _ in range(len(self.model.reactions))]

    def get_sol(self, **kwargs):
        new_prob = self.setup_problem(S=kwargs.get("S") if kwargs.get("S") else self.S,
                                      v_lbs=kwargs.get("lbs") if kwargs.get("lbs") else self.long_lbs,
                                      v_ubs=kwargs.get("ubs") if kwargs.get("ubs") else self.long_ubs,
                                      b=kwargs.get("b") if kwargs.get("b") else self.long_b,
                                      csense=kwargs.get("csense") if kwargs.get("csense") else self.long_csense,
                                      objs=kwargs.get("objs", default=self.long_objs),
                                      row_names=kwargs.get("row_names"),
                                      col_names=kwargs.get("col_names"))
        return new_prob.optimize()

    def get_fva_min_max(self,
                        obj_rxn_id: str) -> (cobra.Solution, cobra.Solution):
        sol_min, sol_max = None, None
        with self.model as model:
            for i, r in model.reactions:
                if r.objective_coefficient != 0:
                    r.lower_bound = self.obj_val_0
                elif r == obj_rxn_id:
                    r.objective_coefficient = 1
                    sol_min = model.optimize(objective_sense="minimize", raise_error=True)
                    sol_max = model.optimize(objective_sense="maximize", raise_error=True)
        return sol_min, sol_max

    def try_estimate(self,
                     lbs: np.ndarray,
                     ubs: np.ndarray,
                     ob_s: np.ndarray,
                     obj_index: int,
                     obj_val: float,
                     fluxes: np.ndarray,
                     status: int,
                     max_iter: int = 50) -> (float, np.ndarray, int):
        lb_c, ub_c = lbs.copy(), ubs.copy()
        for _ in range(max_iter):
            if status == 1 and obj_val > 0:
                return
            # relax the constraint
            lb_c = np.where(lb_c != self.ori_lbs, lb_c - self.m_threshold, lb_c)
            ub_c = np.where(ub_c != self.ori_ubs, ub_c - self.m_threshold, ub_c)
            dx_dt = np.concatenate([np.zeros(shape=(self.S.shape[0],)), lb_c, ub_c])
            prob = self.setup_problem(objs=-ob_s, S=self.A, b=dx_dt,
                                      v_lbs=self.long_lbs, v_ubs=self.long_ubs,
                                      csense=self.long_csense)
            prob_sol = prob.optimize(objective_sense="minimize")
            fluxes = prob_sol.fluxes.copy()
            obj_val = prob_sol.objective_value
            status = 1 if prob_sol.status == "optimal" else 0
        prob = self.setup_problem(objs=self.ori_objs, S=self.S, b=self.ori_b,
                                  v_lbs=lbs, v_ubs=ubs,
                                  csense=self.ori_c)
        prob_sol = prob.optimize(objective_sense="maximize")
        obj_val = -prob_sol.objective_value
        fluxes[obj_index] = -prob_sol.objective_value
        return obj_val, fluxes, status

    def _scan_reactions(self,
                        model,
                        lbs,
                        ubs,
                        all_regs,
                        targets,
                        threshold
                        ) -> (Dict[str, set], np.ndarray):
        tar_related_rxns_dic = {} # gene: set(rxns)
        constrained_rxns = set()
        for i, r in enumerate(model.reactions):
            genes = set([g.id for g in r.genes])
            if len(genes & all_regs) > 0:
                lbs[i] = -threshold if r.lower_bound != 0 else 0
                ubs[i] = threshold if r.upper_bound != 0 else 0
                constrained_rxns.add(i)  # get constrained rxns (line 247
            for t in targets:
                if t in genes:
                    if t not in tar_related_rxns_dic:
                        tar_related_rxns_dic[t] = {i}
                    else:
                        tar_related_rxns_dic[t].add(i)
        constrained_rxns = np.array(list(constrained_rxns))
        return tar_related_rxns_dic, constrained_rxns

    def ko_regulator(self,
                     lbs: np.ndarray,
                     ubs: np.ndarray,
                     regulator: str,
                     targets: Sequence[str],
                     prob_df: pd.DataFrame,
                     reg_id: int,
                     threshold: float,
                     fva_threshold: int):
        all_regs = {regulator}
        with self.model as model:
            ob_s = self.long_objs.copy()
            tar_related_rxns_dic, constrained_rxns = self._scan_reactions(model, lbs, ubs, all_regs, targets, threshold)
            for target in targets:
                for t_r in tar_related_rxns_dic[target]:
                    prob = prob_df[(prob_df["target"] == target) & (prob_df["regulator"] == regulator)]["probability"]
                    if t_r in constrained_rxns and prob < 1:
                        t_g = self.rxn_nums[target]
                        if prob > 0 and self.fluxes_bounds[reg_id] == 0:
                            if model.reactions < fva_threshold:
                                # if model's size is small enough -> do fva
                                self.sol_min, self.sol_max = self.get_fva_min_max(self.rxn_ids[t_r])
                                self.fva_min = self.sol_min.to_frame().iloc[t_g]["fluxes"]
                                self.fva_max = self.sol_max.to_frame().iloc[t_g]["fluxes"]
                            # modify fva flux bounds (line 289
                            if self.fluxes_0[t_g] < 0:
                                self.fluxes_bounds[t_g] = min(self.fva_min,
                                                              self.fva_max,
                                                              self.fluxes_0[t_g])
                            elif self.fluxes_0[t_g] > 0:
                                self.fluxes_bounds[t_g] = max(self.fva_min,
                                                              self.fva_max,
                                                              self.fluxes_0[t_g])
                            else:
                                self.fluxes_bounds[t_g] = max(abs(self.sol_min.to_frame().iloc[t_g]["fluxes"]),
                                                              abs(self.sol_max.to_frame().iloc[t_g]["fluxes"]),
                                                              abs(self.fluxes_0[t_g]))
                        xx = self.fluxes_bounds[t_g] * prob
                        vv = max(abs(self.fluxes_bounds[t_g]), self.m_threshold)
                        if self.fluxes_0[t_g] < 0:
                            # try applying a new constraint
                            temp = max(lbs[t_g], self.ori_lbs[t_g], xx)
                            lbs[t_g] = min(temp, -self.threshold)
                            self.long_ubs[len(ubs) + t_g] = 1000
                            ob_s[len(ubs) + t_g] = -abs(self.obj_val_0) * self.kappa / max(abs(vv),
                                                                                           abs(self.fluxes_bounds[t_g]))
                        elif self.fluxes_0[t_g] > 0:
                            temp = min(ubs[t_g], self.ori_ubs[t_g], xx)
                            ubs[t_g] = max(temp, self.threshold)
                            self.long_ubs[2 * len(ubs) + t_g] = 1000
                            ob_s[2 * len(ubs) + t_g] = min(-abs(self.obj_val_0) * self.kappa / abs(vv),
                                                           ob_s[2 * len(ubs) + t_g])
            dx_dt = np.concatenate([np.zeros(shape=(len(self.model.metabolites,))), lbs, ubs])
            prob = self.setup_problem(objs=ob_s, S=self.A, b=dx_dt, v_lbs=self.long_lbs, v_ubs=self.long_ubs,
                                      csense=self.long_csense)
            prob_sol = prob.optimize(objective_sense="maximize")
            fluxes = prob_sol.fluxes
            obj_val = prob_sol.objective_value
            status = 1 if prob_sol.status == "optimal" else 0
            if status != 1:
                obj_val, fluxes, status = self.try_estimate(lbs, ubs, ob_s,
                                                            obj_index=self.ori_objs.nonzero(),
                                                            max_iter=50,
                                                            obj_val=obj_val,
                                                            status=status,
                                                            fluxes=fluxes)
            ko_sol = self.get_sol(lbs=lbs, ubs=ubs)
        return obj_val, -ko_sol.objective_value, ko_sol.fluxes, fluxes, status

    def prom_iter(self,
                  prob_df: pd.DataFrame):
        self.long_lbs, self.long_ubs = _get_long_lb_ub(len(self.ori_lbs))
        regulators = prob_df.iloc[:, 0].unique().values
        self.regulators = regulators
        obj_vals_wt = np.zeros(len(regulators))
        obj_vals_ko = np.zeros(len(regulators))
        fluxes_ko = np.zeros((len(regulators), len(self.ori_lbs)))
        fluxes_wt = np.zeros((len(regulators), len(self.long_lbs)))
        self.status = np.zeros(len(regulators))
        self.fluxes_bounds = np.zeros((len(regulators),))
        for i, regulator in enumerate(regulators):
            targets = prob_df[prob_df.iloc[:, 0] == regulator].iloc[:, 1].values
            ko_result = self.ko_regulator(self.ori_lbs.copy(),
                                          self.ori_ubs.copy(),
                                          regulator,
                                          targets,
                                          prob_df,
                                          reg_id=i,
                                          threshold=self.threshold,
                                          fva_threshold=1000)
            obj_vals_wt[i], obj_vals_ko[i], fluxes_ko[i, :], fluxes_wt[i, :], self.status[i] = ko_result
        return obj_vals_wt, obj_vals_ko, fluxes_wt, fluxes_ko

    def run(self,
            prob_df: pd.DataFrame):
        fva_sol = flux_variability_analysis(self.model)
        self.fva_min, self.fva_max = fva_sol["minimum"], fva_sol["maximum"]
        with self.model:
            for r in self.model.reactions:
                if r.lower_bound == r.upper_bound:
                    r.lower_bound = r.upper_bound - self.threshold
            sol = self.model.optimize("maximize")
        sol_df = sol.to_frame()
        sol_df[abs(sol_df) < self.threshold] = 0
        self.fluxes_0, self.obj_val_0 = sol_df["fluxes"], sol.objective_value  # might not be right
        return self.prom_iter(prob_df)

    def apply_constr(self):
        pass
