import pandas as pd
import scikit_posthocs as sp
import pingouin as pg
from scipy import stats
import numpy as np
import itertools
from functools import reduce
from .results import NormalityTestResult


class AssumptionTester:
    def __init__(self):
        pass


class NormalityTester(AssumptionTester):
    def __init__(self):
        super(NormalityTester, self).__init__()

    @staticmethod
    def test(data, method="shapiro", **kwargs):
        assert method in ["shapiro", "normaltest", "kstest", "anderson"]

        statistic, pvalue = getattr(stats, method)(data)
        new_result = NormalityTestResult(log={"method": method})
        new_result.add_result(dict(pvalue=pvalue,
                                   statistic=statistic,
                                   data=data,
                                   **kwargs))  # annotations or factors
        return new_result


class HomoscedasticityTester(AssumptionTester):
    def __init__(self):
        super(HomoscedasticityTester, self).__init__()


class MultipleComparisonTester:
    def __init__(self):
        self._assumption_testers = {"normality": NormalityTester(),
                                    "var_homogeneity": HomoscedasticityTester()}
        self._alpha_list = [0.05, 0.01, 0.001, 0.0001]

    def test_assumptions(self,
                         data,
                         test_normality=True,
                         test_var_homogeneity=True,
                         normality_kws=None,
                         var_hom_kws=None):
        if test_normality:
            if normality_kws is None:
                normality_kws = {}
            norm_result = self._assumption_testers["normality"].test(data=data, **normality_kws)

    def test(self,
             data,
             non_parametric=True):
        pass


class PostHocTester:
    def __init__(self):
        pass


class HyperGeometricTester:
    def __init__(self):
        pass





class MultipleComparison:
    def __init__(self,
                 group_data_dict):
        self.data = group_data_dict  # {"group": {"rxn_id": [values]}}
        self.rxn_id_pool = reduce(set.union, [set(rxns.keys()) for grp, rxns in self.data.items()])
        self._results = {}

    def get_result(self, grp1, grp2, rxn_id):
        ordered_groups = sorted([grp1, grp2])
        return self._results[ordered_groups[0]][ordered_groups[1]][rxn_id]

    def get_significant_results(self, sig_threshold=0.05, log_fc_threshold=1.5, dtype="list"):
        if dtype == "list":
            returned_data = []
            for grp, dic in self._results.items():
                for grp_2, dic_2 in dic.items():
                    for rxn_id, vals in dic_2.items():
                        if vals[0] is None:
                            continue
                        if vals[1] <= sig_threshold and abs(vals[2]) >= log_fc_threshold:
                            returned_data.append({"Group 1": grp,
                                                  "Group 2": grp_2,
                                                  "Rxn ID": rxn_id,
                                                  "t": vals[0],
                                                  "p-value": vals[1],
                                                  "log2FC": vals[2]})
        else:
            raise ValueError("dtype is not a valid data type")
        return returned_data

    def anova_fit(self):
        pass

    def cal_all_t_test_results(self):
        for r in self.rxn_id_pool:
            self.t_test_all_grp(r)

    def t_test_all_grp(self, rxn_id):
        comb_iter = itertools.combinations(list(self.data.keys()), 2)
        for grp1, grp2 in comb_iter:
            ordered_groups = sorted([grp1, grp2])
            if ordered_groups[0] not in self._results:
                self._results[ordered_groups[0]] = {}
            if ordered_groups[1] not in self._results[ordered_groups[0]]:
                self._results[ordered_groups[0]][ordered_groups[1]] = {}
            self._results[ordered_groups[0]][ordered_groups[1]][rxn_id] = self.t_test_ind(ordered_groups[0],
                                                                                          ordered_groups[1],
                                                                                          rxn_id)

    def t_test_ind(self, group_1, group_2, rxn_id) -> (float, float, float):
        if rxn_id in self.data[group_1] and rxn_id in self.data[group_2]:
            log_fc = np.log2(np.mean(self.data[group_1][rxn_id]) / np.mean(self.data[group_2][rxn_id]))
            tup = stats.ttest_ind(self.data[group_1][rxn_id], self.data[group_2][rxn_id])
            return tup[0], tup[1], log_fc
        return None, None, None


class StatisticAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.alpha_list = [0.05, 0.01, 0.001, 0.0001]

    def normality_test(self, df_labels='all'):
        if df_labels == 'all':
            df_labels = list(self.df.columns.to_list())

        return {label: stats.shapiro(self.df.loc[:, label]) for label in df_labels}

    def homogeneity_of_var_test(self, col, df_labels='all'):
        pass

    def kruskal_test(self, df_labels='all'):
        if df_labels == 'all':
            df_labels = list(self.df.columns.to_list())

        args = list(map(np.asarray, (self.df[label].values for label in df_labels)))
        alldata = np.concatenate(args)
        if stats.tiecorrect(stats.rankdata(alldata)) == 0:
            return -1, 1  # all values are the same

        return stats.kruskal(*(self.df[label] for label in df_labels))

    def post_hocs(self, df_labels='all', method='wilcoxon'):
        if not hasattr(sp, f'posthoc_{method}'):
            raise ValueError(f'Cannot find {method} test in scikit-posthocs package')
        if df_labels == 'all':
            df_labels = list(self.df.columns.to_list())
        piv_df = self.df[df_labels].melt(var_name="group")
        return getattr(sp, f'posthoc_{method}')(piv_df, val_col="value", group_col='group')