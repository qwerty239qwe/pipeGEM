"""Automated statistical testing workflow.

Provides :class:`AutoStatTest` which automatically selects the
appropriate test based on data characteristics (normality,
homoscedasticity, number of groups) and produces publication-ready
output.
"""
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg

from pipeGEM._logging import get_logger
from pipeGEM.analysis._utils import adjust_p_values
from pipeGEM.analysis.results.auto_stat import AutoStatResult

logger = get_logger(__name__)


def _normalize_pingouin_p_columns(result_df):
    """Keep pingouin p-value column names stable across releases."""
    return result_df.rename(columns={"p_unc": "p-unc"})


# -----------------------------------------------------------------------
# Effect-size helpers
# -----------------------------------------------------------------------

def _cohens_d(group_a, group_b):
    """Cohen's d for two independent groups."""
    na, nb = len(group_a), len(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group_a) - np.mean(group_b)) / pooled_std


def _eta_squared(ss_between, ss_total):
    """Eta-squared effect size."""
    if ss_total == 0:
        return 0.0
    return ss_between / ss_total


# -----------------------------------------------------------------------
# AutoStatTest
# -----------------------------------------------------------------------

class AutoStatTest:
    """Guided statistical test that auto-selects appropriate methods.

    Workflow
    --------
    1. Check sample sizes per group.
    2. Test normality (Shapiro-Wilk for *n* < 50, Kolmogorov-Smirnov
       for *n* >= 50).
    3. Test homoscedasticity (Levene).
    4. Auto-select:
       - 2 groups, normal & equal var -> independent *t*-test
       - 2 groups, non-normal -> Mann-Whitney *U*
       - 3+ groups, normal & equal var -> one-way ANOVA + Tukey HSD
       - 3+ groups, non-normal -> Kruskal-Wallis + Dunn
    5. Apply p-value correction.
    6. Calculate effect sizes (Cohen's *d*, eta-squared).

    Parameters
    ----------
    alpha : float
        Significance level for assumption tests.
    correction : str
        Multiple-comparison correction method (``"bh"``, ``"bonferroni"``,
        ``"holm"``).
    effect_size : bool
        Whether to compute effect sizes.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        correction: str = "bh",
        effect_size: bool = True,
    ):
        self.alpha = alpha
        self.correction = correction
        self.compute_effect_size = effect_size

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def test(
        self,
        data: pd.DataFrame,
        dv: str,
        between: str,
    ) -> AutoStatResult:
        """Run the full guided statistical testing workflow.

        Parameters
        ----------
        data : pandas.DataFrame
            Long-format data.
        dv : str
            Column name of the dependent variable (numeric).
        between : str
            Column name of the grouping variable.

        Returns
        -------
        AutoStatResult
        """
        groups = data[between].unique()
        n_groups = len(groups)
        group_data = {g: data.loc[data[between] == g, dv].dropna().values for g in groups}
        sample_sizes = {g: len(v) for g, v in group_data.items()}

        logger.info("AutoStatTest: %d groups, sizes=%s", n_groups, sample_sizes)

        # ---- Assumption tests ----
        normality = self._check_normality(group_data)
        homoscedasticity = self._check_homoscedasticity(group_data)

        is_normal = all(normality["is_normal"].values())
        is_equal_var = homoscedasticity["is_equal_var"]
        parametric = is_normal and is_equal_var

        logger.info(
            "Assumptions: normal=%s, equal_var=%s -> parametric=%s",
            is_normal, is_equal_var, parametric,
        )

        # ---- Main test ----
        if n_groups == 2:
            main_result = self._two_group_test(data, dv, between, parametric)
        else:
            main_result = self._multi_group_test(data, dv, between, parametric)

        # ---- Pairwise tests (if 3+ groups) ----
        pairwise = None
        if n_groups > 2:
            pairwise = self._pairwise_test(data, dv, between, parametric)

        # ---- Effect sizes ----
        effect_sizes = {}
        if self.compute_effect_size:
            effect_sizes = self._calculate_effect_sizes(group_data, n_groups)

        # ---- P-value correction ----
        corrected_p = None
        if pairwise is not None and "p-unc" in pairwise.columns:
            corrected_p = pd.Series(
                adjust_p_values(pairwise["p-unc"].values, self.correction),
                index=pairwise.index,
                name="p-adj",
            )
            pairwise["p-adj"] = corrected_p

        # ---- Build result ----
        result = AutoStatResult(log={
            "alpha": self.alpha,
            "correction": self.correction,
            "n_groups": n_groups,
            "parametric": parametric,
        })
        result.add_result({
            "main_result": main_result,
            "pairwise_result": pairwise,
            "normality": normality,
            "homoscedasticity": homoscedasticity,
            "effect_sizes": effect_sizes,
            "sample_sizes": sample_sizes,
            "corrected_p_values": corrected_p,
        })
        return result

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _check_normality(self, group_data):
        results = {"method": {}, "statistic": {}, "p_value": {}, "is_normal": {}}
        for g, vals in group_data.items():
            n = len(vals)
            if n < 3:
                results["method"][g] = "skipped"
                results["statistic"][g] = np.nan
                results["p_value"][g] = 1.0
                results["is_normal"][g] = True
                continue
            if n < 50:
                stat, p = stats.shapiro(vals)
                results["method"][g] = "shapiro"
            else:
                stat, p = stats.normaltest(vals)
                results["method"][g] = "normaltest"
            results["statistic"][g] = stat
            results["p_value"][g] = p
            results["is_normal"][g] = p > self.alpha
        return results

    def _check_homoscedasticity(self, group_data):
        arrays = [v for v in group_data.values() if len(v) >= 2]
        if len(arrays) < 2:
            return {"method": "skipped", "statistic": np.nan, "p_value": 1.0, "is_equal_var": True}
        stat, p = stats.levene(*arrays)
        return {"method": "levene", "statistic": stat, "p_value": p, "is_equal_var": p > self.alpha}

    def _two_group_test(self, data, dv, between, parametric):
        if parametric:
            result = pg.ttest(
                data.loc[data[between] == data[between].unique()[0], dv],
                data.loc[data[between] == data[between].unique()[1], dv],
            )
            result["test"] = "t-test"
        else:
            result = pg.mwu(
                data.loc[data[between] == data[between].unique()[0], dv],
                data.loc[data[between] == data[between].unique()[1], dv],
            )
            result["test"] = "Mann-Whitney U"
        return _normalize_pingouin_p_columns(result)

    def _multi_group_test(self, data, dv, between, parametric):
        if parametric:
            result = pg.anova(data=data, dv=dv, between=between)
            result["test"] = "ANOVA"
        else:
            result = pg.kruskal(data=data, dv=dv, between=between)
            result["test"] = "Kruskal-Wallis"
        return _normalize_pingouin_p_columns(result)

    def _pairwise_test(self, data, dv, between, parametric):
        if parametric:
            result = pg.pairwise_tukey(data=data, dv=dv, between=between)
        else:
            result = pg.pairwise_tests(
                data=data, dv=dv, between=between, parametric=False,
            )
        return _normalize_pingouin_p_columns(result)

    def _calculate_effect_sizes(self, group_data, n_groups):
        effects = {}
        groups = list(group_data.keys())
        if n_groups == 2:
            effects["cohens_d"] = _cohens_d(group_data[groups[0]], group_data[groups[1]])
        else:
            # Eta-squared from one-way ANOVA decomposition
            all_vals = np.concatenate(list(group_data.values()))
            grand_mean = np.mean(all_vals)
            ss_between = sum(
                len(v) * (np.mean(v) - grand_mean) ** 2 for v in group_data.values()
            )
            ss_total = np.sum((all_vals - grand_mean) ** 2)
            effects["eta_squared"] = _eta_squared(ss_between, ss_total)
        return effects
