import pandas as pd
import scikit_posthocs as sp
import pingouin as pg
from scipy import stats
import numpy as np
import itertools
from functools import reduce, partial
from pipeGEM.analysis.results import NormalityTestResult, VarHomogeneityTestResult, PairwiseTestResult, \
    MultiGroupComparisonTestResult


DEFAULT_SIGS = [0.05, 0.01, 0.001, 0.0001]


class AssumptionTester:
    def __init__(self):
        pass


class NormalityTester(AssumptionTester):
    def __init__(self):
        super(NormalityTester, self).__init__()

    @staticmethod
    def test(data, dv, group=None, method="shapiro", alpha=0.05, **kwargs):
        assert method in ["shapiro", "normaltest", "kstest", "anderson"]
        if group is None:
            statistic, pvalue = getattr(stats, method)(data[dv])
            result_df = pd.DataFrame({"statistic": [statistic], "p-value": [pvalue]})
        else:
            norm_results = {}
            for g in data[group].unique():
                print(g)
                statistic, pvalue = getattr(stats, method)(data[data[group] == g][dv])
                print(statistic)
                norm_results[g] = {"statistic": statistic, "p-value": pvalue}
            result_df = pd.DataFrame(norm_results).T
        print(result_df)
        result_df["normal"] = (result_df["p-value"] > alpha)
        new_result = NormalityTestResult(log={"method": method,
                                              "group": group,
                                              "alpha": alpha})
        new_result.add_result(dict(result_df=result_df,
                                   data=data,
                                   **kwargs))  # annotations or factors
        return new_result


class HomoscedasticityTester(AssumptionTester):
    def __init__(self):
        super(HomoscedasticityTester, self).__init__()

    @staticmethod
    def test(data, dv, group=None, method="levene", alpha=0.05, **kwargs):
        assert method in ["levene", "bartlett"]
        if group is None:
            input_data = data[dv]
        else:
            input_data = data.groupby(group)[dv].apply(list)
        statistic, pvalue = getattr(stats, method)(*input_data)
        result_df = pd.DataFrame({"statistic": [statistic], "p-value": [pvalue]})
        result_df["equal_var"] = (result_df["p-value"] > alpha)
        new_result = VarHomogeneityTestResult(log={"method": method})
        new_result.add_result(dict(result_df=result_df,
                                   **kwargs))  # annotations or factors
        return new_result


class StatisticalTest:
    def __init__(self):
        self._assumption_testers = {"normality": NormalityTester(),
                                    "var_homogeneity": HomoscedasticityTester()}

    def _test_assumptions(self,
                          data,
                          dv,
                          group,
                          test_normality=True,
                          test_homoscedasticity=True,
                          normality_kws=None,
                          homoscedasticity_kws=None):
        test_results = []
        if test_normality:
            if normality_kws is None:
                normality_kws = {}
            test_results.append(self._assumption_testers["normality"].test(data=data, dv=dv, group=group,
                                                                           **normality_kws))
        if test_homoscedasticity:
            if homoscedasticity_kws is None:
                homoscedasticity_kws = {}
            test_results.append(self._assumption_testers["var_homogeneity"].test(data=data, dv=dv, group=group,
                                                                                 **homoscedasticity_kws))

        return test_results

    def _to_use_parametric_test(self, data, dv, group, **kwargs):
        test_results = self._test_assumptions(data, dv, group, **kwargs)
        equal_var, normal = True, True

        for result in test_results:
            if isinstance(result, VarHomogeneityTestResult):
                equal_var = result.result_df["equal_var"].all()
            elif isinstance(result, NormalityTestResult):
                normal = result.result_df["normal"].all()

        return (equal_var and normal), test_results


class PairwiseTester(StatisticalTest):
    def __init__(self):
        super().__init__()
        self._alpha_list = DEFAULT_SIGS
        self.parametric_methods = {"tukey": (pg.pairwise_tukey, "pingouin"),
                                   "dunn": (sp.posthoc_dunn, "scikit_posthocs"),
                                   }
        self.non_parametric_methods = {"mw": (partial(pg.pairwise_tests, parametric=False,
                                                      within=None), "pingouin"),
                                       "wilcoxon": (partial(pg.pairwise_tests,
                                                            parametric=False), "pingouin")}

    def test(self,
             data,
             dep_var,
             between,
             parametric=False,
             method="mw",
             added_label=None,
             parametric_params=None,
             **kwargs):
        result_obj = PairwiseTestResult(dict(dep_var=dep_var,
                                             between=between,
                                             parametric=parametric,
                                             method=method,
                                             **kwargs))
        assump_test_results = None
        if parametric == "auto":
            parametric_params = parametric_params or {}
            parametric, assump_test_results = self._to_use_parametric_test(data=data, dv=dep_var, group=between,
                                                                           **parametric_params)

        method_pool = self.non_parametric_methods if parametric else self.non_parametric_methods
        if method_pool[method][1] == "scikit_posthocs":
            # TODO: fix
            result = method_pool[method][0](data,
                                            val_col=dep_var,
                                            group_col=between,
                                            **kwargs)
            if added_label is not None:
                result["label"] = added_label
            result_obj.add_result(dict(result_df=result,
                                       assump_test_results=assump_test_results))
        elif method_pool[method][1] == "pingouin":
            result = method_pool[method][0](data,
                                            dv=dep_var,
                                            between=between,
                                            **kwargs)
            if added_label is not None:
                result["label"] = added_label
            result_obj.add_result(dict(result_df=result,
                                       p_value_col="p-unc",
                                       assump_test_results=assump_test_results))

        return result_obj


class MultiGroupComparison(StatisticalTest):
    def __init__(self):
        super().__init__()
        self._alpha_list = DEFAULT_SIGS

    def test(self,
             data,
             dep_var,
             between,
             parametric=True,
             parametric_params=None,
             added_label=None,
             **kwargs):
        assump_test_results = None
        result = MultiGroupComparisonTestResult(log=dict(dep_var=dep_var,
                                                         between=between,
                                                         parametric=parametric,
                                                         parametric_params=parametric_params,
                                                         added_label=added_label,
                                                         **kwargs))

        if parametric == "auto":
            parametric_params = parametric_params or {}
            parametric, assump_test_results = self._to_use_parametric_test(data=data, dv=dep_var, group=between,
                                                                           **parametric_params)

        if parametric is True:
            result_df = pg.anova(data=data,
                                 dv=dep_var,
                                 between=between,
                                 **kwargs)
        else:
            result_df = pg.kruskal(data=data,
                                   dv=dep_var,
                                   between=between,
                                   **kwargs)
        if added_label is not None:
            result_df["label"] = added_label

        result.add_result(dict(result_df=result_df,
                               assump_test_results=assump_test_results,
                               inferred_parametric=parametric,
                               p_value_col="p-unc"))
        return result


def oneway_mkw(data, dv, between, **kwargs):
    pass