import pandas as pd
import scikit_posthocs as sp
import pingouin as pg
from scipy import stats
import numpy as np
import itertools
from functools import reduce, partial
from pipeGEM.analysis.results import NormalityTestResult, VarHomogeneityTestResult, PairwiseTestResult


DEFAULT_SIGS = [0.05, 0.01, 0.001, 0.0001]


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

    @staticmethod
    def test(data, method="levene", **kwargs):
        assert method in ["levene", "bartlett"]
        statistic, pvalue = getattr(stats, method)(data)
        new_result = VarHomogeneityTestResult(log={"method": method})
        new_result.add_result(dict(pvalue=pvalue,
                                   statistic=statistic,
                                   data=data,
                                   **kwargs))  # annotations or factors
        return new_result


class PairwiseTester:
    def __init__(self):
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
             **kwargs):
        result_obj = PairwiseTestResult(dict(dep_var=dep_var,
                                             between=between,
                                             parametric=parametric,
                                             method=method,
                                             **kwargs))
        method_pool = self.non_parametric_methods if parametric else self.non_parametric_methods
        if method_pool[method][1] == "scikit_posthocs":
            result = method_pool[method][0](data,
                                            val_col=dep_var,
                                            group_col=between,
                                            **kwargs)
            if added_label is not None:
                result["label"] = added_label
            result_obj.add_result(dict(result_df=result))
        elif method_pool[method][1] == "pingouin":
            result = method_pool[method][0](data,
                                            dv=dep_var,
                                            between=between,
                                            **kwargs)
            if added_label is not None:
                result["label"] = added_label
            result_obj.add_result(dict(result_df=result))

        return result_obj


class MultiGroupComparison:
    def __init__(self):
        self._assumption_testers = {"normality": NormalityTester(),
                                    "var_homogeneity": HomoscedasticityTester()}
        self._alpha_list = DEFAULT_SIGS

    def test_assumptions(self,
                         data,
                         test_normality=True,
                         test_homoscedasticity=True,
                         normality_kws=None,
                         homoscedasticity_kws=None):
        if test_normality:
            if normality_kws is None:
                normality_kws = {}
            return self._assumption_testers["normality"].test(data=data, **normality_kws)
        if test_homoscedasticity:
            if normality_kws is None:
                homoscedasticity_kws = {}
            return self._assumption_testers["var_homogeneity"].test(data=data, **homoscedasticity_kws)

    def test(self,
             data,
             dep_var,
             between,
             parametric=True,
             **kwargs):
        if parametric is True:
            result = pg.anova(data=data,
                              dv=dep_var,
                              between=between,
                              **kwargs)
        else:
            result = pg.kruskal(data=data,
                                dv=dep_var,
                                between=between,
                                **kwargs)


def oneway_mkw(data, dv, between, **kwargs):
    pass



class HyperGeometricTester:
    def __init__(self):
        pass
