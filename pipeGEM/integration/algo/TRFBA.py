import warnings
from typing import Union, Sequence, Dict

import matplotlib.pyplot as plt
from scipy import stats
import cobra
from cobra.flux_analysis import flux_variability_analysis

import pandas as pd
import numpy as np

from pipeGEM.analysis import Problem


def convert_to_without_or(model: cobra.Model):
    model = model.copy()
    new_rs = []
    ori_rs = model.reactions
    for r in ori_rs:
        for i, new_gr in enumerate(r.gene_reaction_rule.split(" or ")):
            new_r = cobra.Reaction(id=f"{r.id}_g_{i}",
                                   name=r.name,
                                   subsystem=r.subsystem,
                                   lower_bound=r.lower_bound,
                                   upper_bound=r.upper_bound)
            new_r.add_metabolites(r.metabolites)
            new_r.gene_reaction_rule = new_gr
            new_rs.append(new_r)
    model.add_reactions(new_rs)
    model.remove_reactions(ori_rs)
    return model


class TRFBA_MILP(Problem):
    def __init__(self, model, data, exp_col_name, regulation_df: pd.DataFrame, C=0.68):
        super().__init__(model)  # should be no_or_model
        self.data = data
        self.exp_col_name = exp_col_name
        self.regulation_df = regulation_df.copy()
        self.regulation_df = regulation_df.merge(self.data, how="outer", left_on="regulator", right_index=True)
        self.C = C
        self.prepare_problem()

    def prepare_problem(self):
        super().prepare_problem()
        gene_in_data = [g.id for g in self.model.genes if g.id in self.data.index and len(g.reactions) > 0]
        append_genes = [g.id for g in self.model.genes if g.id in self.data.index]
        append_genes += [g for g in self.regulation_df["target"] if g not in append_genes and g in self.data.index]
        append_genes += [g for g in self.regulation_df["regulator"] if g not in append_genes and g in self.data.index]

        self.add_S_cols = append_genes
        self.add_S_rows = gene_in_data
        rxn_pos_dic = {r.id: i for i, r in enumerate(self.model.reactions)}
        gen_pos_dic = {g: i for i, g in enumerate(append_genes)}
        self.long_ubs = np.concatenate([self.ori_ubs, np.array([max(self.data.loc[g, :]) for g in append_genes])])
        self.long_lbs = np.concatenate([self.ori_lbs, np.zeros((len(append_genes), ))])
        self.A = np.concatenate([np.concatenate([self.S, np.zeros(self.S.shape[0], len(append_genes))], axis=1),
                                 np.zeros(len(gene_in_data), self.S.shape[1] + len(append_genes))], axis=0)
        self.long_objs = np.concatenate([self.ori_objs, np.zeros((len(append_genes), ))])
        self.long_b = np.concatenate([self.ori_b, np.zeros(len(append_genes),)])
        self.long_csense = np.concatenate([self.ori_c, np.array(["L" for _ in range(len(gene_in_data))])])
        for g_pos, g in enumerate(gene_in_data):
            for gr in g.reactions:
                self.A[self.S.shape[0] + g_pos, rxn_pos_dic[gr.id]] = 1
                self.A[self.S.shape[0] + g_pos, self.S.shape[1] + gen_pos_dic[g]] = self.C
        self.tf_vars = self.get_tf_vars()
        self.x_max_lefts, self.x_max_rights, self.polys, self.targets = self.fit_regulation()
        self.extend_to_MILP(self.x_max_lefts, self.x_max_rights, self.polys, self.targets)

    def get_tf_vars(self):
        reg = self.regulation_df[self.regulation_df["target"].isin(self.data.index)]
        gb = reg.groupby("target")
        reg_exp_vals = gb[self.exp_col_name].sum().apply(lambda x: list(x), axis=1).to_frame()  # Xs
        reg_list = gb["regulator"].apply(lambda x: list(x)).to_frame()
        targets = reg_exp_vals.index.to_frame()
        target_vals = self.data.loc[targets, self.exp_col_name].apply(lambda x: list(x), axis=1).to_frame()  # Ys
        reg_exp_vals.columns = ["reg values"]
        reg_list.columns = ["reg genes"]
        targets.columns = ["target gene"]
        target_vals.columns = ["target values"]
        tf_vars = pd.concat([targets, target_vals, reg_list, reg_exp_vals], axis=1)
        return tf_vars

    def fit_regulation(self, ni=3, plot: bool = False):
        x_max_lefts, x_max_rights = [], []
        polys = []
        targets = []

        for i, tf in self.tf_vars.iterrows():
            x, y = tf["reg values"], tf["target values"]
            if plot:
                plt.plot(x, y)
            cuts = np.quantile(x, [(i+1) / ni for i in range(ni-1)])
            y_max, x_max = [], []
            for k in range(ni):
                x_lower = 0 if k == 0 else cuts[k - 1]
                x_upper = np.inf if k == ni - 1 else cuts[k]
                selected_x = x[x_lower <= x < x_upper]
                selected_y = y[x_lower <= x < x_upper]
                y_max.append(np.max(selected_y))
                x_max.append(selected_x[np.argmax(selected_y)])

            for k in range(ni - 1):
                x_max_lefts.append(x_max[k])
                x_max_rights.append(x_max[k+1])
                polys.append(np.polyfit([x_max[k], x_max[k+1]], [y_max[k], y_max[k+1]], 1))
                targets.append(tf["target gene"])
        return x_max_lefts, x_max_rights, polys, targets

    def extend_to_MILP(self, x_max_lefts, x_max_rights, polys, targets):
        reg_max = self.tf_vars["reg values"].apply(lambda x: max(x)).max()
        x_max_lefts, x_max_rights = x_max_lefts.copy(), x_max_rights.copy()
        x_max_lefts[0] = 0
        x_max_rights[-1] = reg_max * 5

        n_r, n_c = self.A.shape
        ext_A = np.zeros(shape=(len(targets) * 3, n_c + len(targets) * 2))
        ext_A_rhs = np.zeros(shape=(n_r, len(targets) * 2))
        ext_csenses = np.array(["L" for _ in range(len(targets) * 3)])
        ext_bs = np.zeros(shape=(len(targets) * 3,))
        ext_lbs, ext_ubs = np.zeros(shape=(len(targets) * 2,)), np.ones(shape=(len(targets) * 2,))
        for i, (poly, target, x_l, x_r) in enumerate(zip(polys, targets, x_max_lefts, x_max_rights)):
            regs = self.tf_vars.loc[self.tf_vars["target gene"] == target, "reg genes"].values
            reg_pos = n_c + np.array([self.add_S_cols.index(r) for r in regs])
            tar_pos = n_c + self.add_S_cols.index(target)
            ext_A[3 * i, reg_pos] = -poly[0]
            ext_A[3 * i + 1, reg_pos], ext_A[3 * i + 2, reg_pos] = 1, -1
            ext_A[3 * i, tar_pos] = 1
            ext_bs[3 * i] = poly[1]
            ext_bs[3 * i + 1], ext_bs[3 * i + 2] = reg_max + x_l, reg_max + x_r
            ext_A[3 * i, 2 * i], ext_A[3 * i, 2 * i + 1] = -reg_max, -reg_max
            ext_A[3 * i + 1, 2 * i], ext_A[3 * i + 2, 2 * i + 1] = reg_max, reg_max
        self.var_types = np.concatenate([np.array(["C" for _ in range(self.A.shape[1])]),
                                         np.array(["B" for _ in range(len(targets) * 2)])])
        self.A = np.concatenate([self.A, ext_A_rhs], axis=1)
        self.A = np.concatenate([self.A, ext_A], axis=0)
        self.long_csense = np.concatenate([self.long_csense, ext_csenses])
        self.long_b = np.concatenate([self.long_b, ext_bs])
        self.long_lbs = np.concatenate([self.long_lbs, ext_lbs])
        self.long_ubs = np.concatenate([self.long_ubs, ext_ubs])


class TRFBA:
    regulators_loc = 0
    targets_loc = 1

    def __init__(self, model, regulation_df):
        self.model = model
        self.regulation_df = regulation_df

    def _check_genes(self, raise_error=False):
        gene_id = [g.id for g in self.model.genes]
        targets = self.regulation_df.iloc[:, self.targets_loc]
        if raise_error:
            for target in targets:
                if target not in gene_id:
                    raise ValueError(f"A target is not shown in model's gene list: {target}")
        else:
            self.regulation_df = self.regulation_df[self.regulation_df.iloc[:, 1].isin(targets)]
