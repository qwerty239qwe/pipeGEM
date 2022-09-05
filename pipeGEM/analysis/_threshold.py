import warnings

from typing import Union

import pandas as pd
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
import numpy as np
from pipeGEM.utils import ObjectFactory
from pipeGEM.analysis import rFastCormicThresholdAnalysis, PercentileThresholdAnalysis, LocalThresholdAnalysis, timing


class ThresholdFinders(ObjectFactory):
    def __init__(self):
        super(ThresholdFinders, self).__init__()


class ThresholdFinder:
    def __init__(self):
        pass

    def find_threshold(self, **kwargs):
        raise NotImplementedError("")


class DistributionBased(ThresholdFinder):
    def __init__(self):
        super().__init__()

    def find_threshold(self, **kwargs):
        raise NotImplementedError("")

    @staticmethod
    def gaussian_dist(x, amp, cen, wid):
        return amp * np.exp(-np.power(x - cen, 2) / wid)

    def bimodal_dist(self, x, A1, mu1, wid1, A2, mu2, wid2):
        return self.gaussian_dist(x, A1, mu1, wid1) + self.gaussian_dist(x, A2, mu2, wid2)

    @staticmethod
    def _get_second_deriv(y, dx):
        return np.gradient(np.gradient(y, dx), dx)

    @staticmethod
    def _cal_canyons_p(x1, y1, x2, y2, c=0.5):
        return c * 2 ** (-abs(x1 - x2)) + (1 - c) * 1.5 ** (((y1 / y2) if y1 > y2 else (y2 / y1)) - 1) / 50

    def get_init_for_bimodal(self, x, y, min_h_ratio=1.5, max_w_ratio=2, n_top=100):
        assert len(x) == len(y)

        dx = x[1] - x[0]
        ypp = self._get_second_deriv(y, dx)
        peak = np.max(y)
        xmin, xmax = x[y > np.max(y) * 0.1][0], x[y > np.max(y) * 0.1][-1]

        prev = None
        cans = []
        can_deeps = []

        is_inc = False
        for i, yi in enumerate(ypp):
            if prev is None:
                prev = yi
            else:
                grad = yi - prev
                if grad > 0:
                    if not is_inc:
                        cans.append(i - 1)
                        can_deeps.append(prev)
                        is_inc = True
                else:
                    is_inc = False
                prev = yi

        candidates = x[np.array(cans)[np.argsort(can_deeps)]][:n_top]
        y_of_cands = y[np.array(cans)[np.argsort(can_deeps)]][:n_top]
        # grid search
        p_arr = np.ones(shape=(len(candidates), len(y_of_cands))) * np.inf
        for i, (x1, y1) in enumerate(zip(candidates, y_of_cands)):
            for j, (x2, y2) in enumerate(zip(candidates, y_of_cands)):
                is_invalid_h = min_h_ratio * y1 < peak and min_h_ratio * y2 < peak
                is_invalid_w = ((min(x1, x2) - xmin) / (xmax - max(x1, x2)) > max_w_ratio) or \
                               ((min(x1, x2) - xmin) / (xmax - max(x1, x2)) < 1 / max_w_ratio)
                if i >= j or is_invalid_h or is_invalid_w:
                    continue
                p_arr[i, j] = self._cal_canyons_p(x1=x1, y1=y1, x2=x2, y2=y2)
        arg_min = np.where(p_arr == np.min(p_arr))
        return candidates[arg_min[0][0]], candidates[arg_min[1][0]]

    @staticmethod
    def _check_fitted_param(p, amp_ratio_tol=4, var_ratio_tol=2, mean_diff_tol=4, verbosity=False):
        amp1, mean1, cov1, amp2, mean2, cov2 = p
        if abs(np.log2(amp1) - np.log2(amp2)) > np.log2(amp_ratio_tol):
            if verbosity:
                print("Two gaussian have too large difference in amp")
            return False
        if abs(np.log2(cov1) - np.log2(cov1)) > np.log2(var_ratio_tol):
            if verbosity:
                print("Two gaussian have too large difference in var")
            return False
        if abs(mean1 - mean2) < mean_diff_tol:
            if verbosity:
                print("Two gaussian have too small mean distance")
            return False
        return True

    @staticmethod
    def _get_y_by_nearest_x(x, y, c):
        dtx = abs(x - c)
        return y[np.argsort(dtx)[0]]

    def _bimodal_fit(self, x, y, amp_ratio_tol=4, var_ratio_tol=6, mean_diff_tol=3, return_best_p=True):
        c1, c2 = self.get_init_for_bimodal(x, y)
        c1, c2 = min(c1, c2), max(c1, c2)
        print("original guess: ", c1, c2)
        init_vals = (1, c1, 1, 1, c2, 1)
        init_val_displayed = (self._get_y_by_nearest_x(x, y, c1), c1, 1, self._get_y_by_nearest_x(x, y, c2), c2, 1)
        grid = [(2 + 10 * i, 2 + 10 * j) for i in range(0, 10) for j in range(0, 10)]
        covar = None
        p = init_val_displayed
        try:
            found_best = False
            best_pscore = np.inf
            best_p = None
            it = 0
            while (not found_best) and it < len(grid):
                tried_vals = [grid[it][0], init_vals[1], init_vals[2],
                              grid[it][1], init_vals[4], init_vals[5]]
                p, covar = curve_fit(self.bimodal_dist, xdata=x, ydata=y,
                                     p0=tried_vals, bounds=((0, -np.inf, 0, 0, -np.inf, 0), np.inf))
                found_best = self._check_fitted_param(p,
                                                      amp_ratio_tol=amp_ratio_tol,
                                                      var_ratio_tol=var_ratio_tol,
                                                      mean_diff_tol=mean_diff_tol)
                pscore = self._cal_canyons_p(x1=p[1], x2=p[4], y1=p[0], y2=p[3])
                if best_pscore > pscore:
                    best_p = p
                it += 1
            if not found_best:
                if return_best_p:
                    warnings.warn("Fail to find proper parameters, use the best one")
                    p = (i for i in best_p)
                else:
                    warnings.warn("Fail to find proper parameters, use initial values")
                    print("problematic p: ", best_p)
                    self._check_fitted_param(p,
                                             amp_ratio_tol=amp_ratio_tol,
                                             var_ratio_tol=var_ratio_tol,
                                             mean_diff_tol=mean_diff_tol, verbosity=True)
                    p = init_val_displayed
                    covar = None
        except RuntimeError as e:
            warnings.warn(f"Fail to optimize: {e}")
        A1, mu1, wid1, A2, mu2, wid2 = p
        print("fitted Amps: ", A1, A2)
        print("fitted means: ", mu1, mu2)
        if mu1 < mu2:
            return (A2, mu2, wid2,), (A1, mu1, wid1,), covar
        return (A1, mu1, wid1,), (A2, mu2, wid2,), covar


class rFastCormicsThreshold(DistributionBased):

    def __init__(self):
        super().__init__()

    @timing
    def find_threshold(self,
                       data: Union[np.ndarray, pd.Series, dict],
                       cut_off: float = -np.inf,
                       use_first_guess: bool = False):
        if isinstance(data, pd.Series):
            arr = data.values
        elif isinstance(data, dict):
            arr = np.array(list(data.values()))
        else:
            arr = data

        arr = arr[arr > cut_off]
        kde_f = gaussian_kde(arr)
        x = np.linspace(arr.min(), arr.max(), 10000)
        y = kde_f(x)
        if use_first_guess:
            c1, c2 = self.get_init_for_bimodal(x, y)
            c1, c2 = min(c1, c2), max(c1, c2)
            best_vals_right, best_vals_left = (1, c2, 1), (1, c1, 1)
        else:
            best_vals_right, best_vals_left, _ = self._bimodal_fit(x, y)

        result = rFastCormicThresholdAnalysis(log={"use_first_guess": use_first_guess, "cut_off": cut_off})
        right_c = None if use_first_guess else self.gaussian_dist(x, *tuple(best_vals_right))
        left_c = None if use_first_guess else self.gaussian_dist(x, *tuple(best_vals_left))
        result.add_result(exp_th=best_vals_right[1], nonexp_th=best_vals_left[1],
                          right_curve=right_c, left_curve=left_c, x=x, y=y)

        return result


class RankBased(ThresholdFinder):
    def __init__(self):
        super().__init__()

    def find_threshold(self, **kwargs):
        raise NotImplementedError()


class PercentileThreshold(RankBased):
    def __init__(self):
        super().__init__()

    def find_threshold(self,
                       data,  # 1d array
                       p,
                       **kwargs):
        assert 0 <= p <= 100
        if isinstance(data, pd.Series):
            arr = data.values
        elif isinstance(data, dict):
            arr = np.array(list(data.values()))
        else:
            arr = data
        arr = arr[np.isfinite(arr)]
        result = PercentileThresholdAnalysis(log={"p": p})
        exp_th = np.percentile(arr, q=p)
        result.add_result(data=arr, exp_th=exp_th)
        return result


class LocalThreshold(RankBased):
    def __init__(self):
        super().__init__()

    def find_threshold(self,
                       data,  # 2d array
                       p,
                       **kwargs):
        assert 0 <= p <= 100
        if isinstance(data, pd.DataFrame):
            arr = data.values
            genes = data.index
        else:
            arr = data
            genes = kwargs.get("genes")
        arr[~np.isfinite(arr)] = np.nan
        exp_ths = pd.DataFrame({"exp_th": np.nanpercentile(arr, q=p, axis=1)}, index=genes)
        result = LocalThresholdAnalysis(log={"p": p})
        result.add_result(exp_ths)
        return result


threshold_finders = ThresholdFinders()
threshold_finders.register("rFASTCORMICS", rFastCormicsThreshold)
threshold_finders.register("percentile", PercentileThreshold)
