import warnings

from typing import Union, List, Optional, Tuple, Dict

import pandas as pd
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
import numpy as np
from pipeGEM.utils import ObjectFactory
from pipeGEM.analysis.results.thresholds import rFASTCORMICSThresholdAnalysis, PercentileThresholdAnalysis, LocalThresholdAnalysis, timing


class ThresholdFinders(ObjectFactory):
    """Factory class for registering and creating threshold finding algorithms."""
    def __init__(self):
        """Initializes the ThresholdFinders factory."""
        super(ThresholdFinders, self).__init__()


class ThresholdFinder:
    """Abstract base class for threshold finding algorithms."""
    def __init__(self):
        """Initializes the ThresholdFinder."""
        pass

    def find_threshold(self, **kwargs):
        """
        Abstract method to find expression thresholds.

        Subclasses must implement this method.

        Parameters
        ----------
        **kwargs
            Algorithm-specific parameters.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("")


class DistributionBased(ThresholdFinder):
    """Base class for threshold finders based on data distribution analysis."""
    def __init__(self):
        """Initializes the DistributionBased finder."""
        super().__init__()

    def find_threshold(self, **kwargs):
        """
        Abstract method for distribution-based threshold finding.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("")

    @staticmethod
    def gaussian_dist(x: np.ndarray, amp: float, cen: float, wid: float) -> np.ndarray:
        """
        Calculate the Gaussian function value.

        Parameters
        ----------
        x : np.ndarray
            Input array.
        amp : float
            Amplitude (height) of the Gaussian peak.
        cen : float
            Center (mean) of the Gaussian distribution.
        wid : float
            Width (related to variance) of the Gaussian distribution.

        Returns
        -------
        np.ndarray
            Gaussian function values corresponding to x.
        """
        # Note: 'wid' here might represent 2*sigma^2 depending on convention.
        return amp * np.exp(-np.power(x - cen, 2) / wid)

    def bimodal_dist(self, x: np.ndarray, A1: float, mu1: float, wid1: float, A2: float, mu2: float, wid2: float) -> np.ndarray:
        """
        Calculate the sum of two Gaussian distributions (bimodal).

        Parameters
        ----------
        x : np.ndarray
            Input array.
        A1, mu1, wid1 : float
            Amplitude, center, and width of the first Gaussian component.
        A2, mu2, wid2 : float
            Amplitude, center, and width of the second Gaussian component.

        Returns
        -------
        np.ndarray
            Sum of the two Gaussian functions evaluated at x.
        """
        return self.gaussian_dist(x, A1, mu1, wid1) + self.gaussian_dist(x, A2, mu2, wid2)

    @staticmethod
    def _get_second_deriv(y, dx):
        return np.gradient(np.gradient(y, dx), dx)

    @staticmethod
    def _cal_canyons_p(x1, y1, x2, y2, c=0.5):
        return c * 2 ** (-abs(x1 - x2)) + (1 - c) * 1.5 ** (((y1 / y2) if y1 > y2 else (y2 / y1)) - 1) / 50

    def get_init_for_bimodal(self, x: np.ndarray, y: np.ndarray, max_x: float, min_x: float,
                             min_h_ratio: float = 1.5, max_w_ratio: float = 2, n_top: int = 100) -> Tuple[float, float]:
        """
        Heuristically find initial guesses for the centers of two modes in a distribution.

        Analyzes the second derivative of the distribution `y` with respect to `x`
        to find potential peaks (local minima in the second derivative) and selects
        the best pair based on height, width ratios, and a scoring function.

        Parameters
        ----------
        x : np.ndarray
            The x-coordinates of the distribution data points. Assumed to be evenly spaced.
        y : np.ndarray
            The y-coordinates (density/frequency) of the distribution data points.
        max_x : float
            Maximum allowed x-value for candidate peaks.
        min_x : float
            Minimum allowed x-value for candidate peaks.
        min_h_ratio : float, optional
            Minimum ratio relative to the global peak height for a candidate peak
            to be considered valid. Default is 1.5.
        max_w_ratio : float, optional
            Maximum allowed ratio between the widths defined by the two candidate
            peaks relative to the overall data range. Default is 2.
        n_top : int, optional
            Number of top candidate peaks (based on second derivative depth) to consider.
            Default is 100.

        Returns
        -------
        Tuple[float, float]
            A tuple containing the estimated x-coordinates (centers) of the two modes.
        """
        assert len(x) == len(y), "x and y must have the same length."

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
                if i >= j or is_invalid_h or is_invalid_w or x1 > max_x or x2 > max_x or x1 < min_x or x2 < min_x:
                    continue
                p_arr[i, j] = self._cal_canyons_p(x1=x1, y1=y1, x2=x2, y2=y2)
        arg_min = np.where(p_arr == np.min(p_arr))
        print("p_score of init values:", np.min(p_arr))
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

    def _bimodal_fit(self, x, y, max_x, min_x,
                     amp_ratio_tol=4, var_ratio_tol=6, mean_diff_tol=3,
                     return_heuristic=False, k=1):
        c1, c2 = self.get_init_for_bimodal(x, y, max_x=max_x, min_x=min_x)
        c1, c2 = min(c1, c2), max(c1, c2)
        print("original guess: ", c1, c2)
        init_vals = (1, c1, 1, 1, c2, 1)
        init_val_displayed = (self._get_y_by_nearest_x(x, y, c1), c1, 1, self._get_y_by_nearest_x(x, y, c2), c2, 1)
        grid = [(2 + 20 * i, 2 + 20 * j) for i in range(0, 5) for j in range(0, 5)]
        covar = None
        pscore_arr = np.ones(shape=(k,)) * np.inf
        params_arr = np.empty(shape=(k, 6))
        params_arr[:] = np.nan

        p = init_val_displayed
        try:
            found_best = False
            # best_pscore = np.inf
            # best_p = None
            it = 0
            while it < len(grid):
                tried_vals = [grid[it][0], init_vals[1], init_vals[2],
                              grid[it][1], init_vals[4], init_vals[5]]
                x_var = (max_x - min_x) / 6
                p, covar = curve_fit(self.bimodal_dist, xdata=x, ydata=y,
                                     p0=tried_vals, bounds=((0, max(min_x, init_vals[1] - x_var), 0,
                                                             0, max(min_x, init_vals[4] - x_var), 0),
                                                            (np.inf, min(max_x, init_vals[1] + x_var), np.inf,
                                                             np.inf, min(max_x, init_vals[4] + x_var), np.inf)))
                pscore = np.inf
                if (min_x < p[1] < max_x) and (min_x < p[4] < max_x):  # valid x (in acceptable range)
                    found_best = self._check_fitted_param(p,
                                                          amp_ratio_tol=amp_ratio_tol,
                                                          var_ratio_tol=var_ratio_tol,
                                                          mean_diff_tol=mean_diff_tol)
                    pscore = self._cal_canyons_p(x1=p[1], x2=p[4], y1=p[0], y2=p[3])
                    # print("pscore:", pscore)
                insert_index = np.searchsorted(pscore_arr, pscore)
                if insert_index < k:
                    pscore_arr = np.insert(pscore_arr, insert_index, pscore)[:k]
                    params_arr = np.insert(params_arr, insert_index, np.array(p), axis=0)[:k, :]
                it += 1

            if not found_best:
                if not return_heuristic:
                    warnings.warn(f"Fail to find proper parameters, return the best {k} params")
                else:
                    warnings.warn("Fail to find proper parameters, use initial values")
                    print("problematic parameter array: ", params_arr)
                    self._check_fitted_param(p,
                                             amp_ratio_tol=amp_ratio_tol,
                                             var_ratio_tol=var_ratio_tol,
                                             mean_diff_tol=mean_diff_tol, verbosity=True)
                    params_arr[:] = init_val_displayed
                    covar = None
        except RuntimeError as e:
            warnings.warn(f"Fail to optimize: {e}")

        # move the larger means to the first 3 columns
        params_arr[params_arr[:, 1] < params_arr[:, 4], :] = params_arr[params_arr[:, 1] < params_arr[:, 4], :][:, [3, 4, 5, 0, 1, 2]]
        A1, A2 = params_arr[0, 0], params_arr[0, 3]
        mu1, mu2 = params_arr[0, 1], params_arr[0, 4]
        print("best fitted Amps: ", A1, A2)
        print("best fitted means: ", mu1, mu2)
        return params_arr, c1, c2


class rFASTCORMICSThreshold(DistributionBased):
    """
    Finds expression thresholds by fitting a bimodal Gaussian distribution.

    This method assumes gene expression data often follows a bimodal distribution,
    representing 'off' and 'on' states. It fits two Gaussian curves to the data's
    Kernel Density Estimate (KDE) and uses the means of these curves as potential
    thresholds.
    """
    def __init__(self):
        """Initializes the rFASTCORMICSThreshold finder."""
        super().__init__()

    @timing
    def find_threshold(self,
                       data: Union[np.ndarray, pd.Series, dict, list],
                       cut_off: float = -np.inf,
                       return_heuristic: bool = False,
                       hard_x_lims: tuple = (0.05, 0.95),
                       k_best: int = 3) -> rFASTCORMICSThresholdAnalysis:
        """
        Finds expression and non-expression thresholds using bimodal Gaussian fitting.

        Filters data below `cut_off`, calculates KDE, fits a bimodal Gaussian,
        and returns the means of the fitted Gaussians as potential thresholds.

        Parameters
        ----------
        data : Union[np.ndarray, pd.Series, dict, list]
            Input expression data (typically log-transformed).
        cut_off : float, optional
            Value below which data points are excluded before analysis.
            Default is -infinity (no cutoff).
        return_heuristic : bool, optional
            If True, returns the initial heuristic guesses for the means instead
            of the fitted means. Default is False.
        hard_x_lims : tuple, optional
            Tuple defining the percentile range (e.g., (0.05, 0.95)) of the data
            to consider for fitting, restricting the search space for means.
            Default is (0.05, 0.95).
        k_best : int, optional
            Number of best-fitting parameter sets to return (ranked by p-score).
            Default is 3.

        Returns
        -------
        rFASTCORMICSThresholdAnalysis
            An analysis object containing the calculated thresholds, fitted curves,
            and intermediate data (x, y KDE values).
        """
        assert 0 <= hard_x_lims[0] < hard_x_lims[1] <= 1, "hard_x_lims must be percentiles between 0 and 1."

        if isinstance(data, pd.Series):
            arr = data.values
        elif isinstance(data, dict):
            arr = np.array(list(data.values()))
        elif isinstance(data, list):
            arr = np.array(data)
        else:
            arr = data

        print(f"cutting off {len(arr[arr <= cut_off])} data since their expression values are below {cut_off}")
        arr = arr[arr > cut_off]
        if arr.shape[0] == 0:
            result = rFASTCORMICSThresholdAnalysis(log={"use_first_guess": return_heuristic,
                                                        "cut_off": cut_off,
                                                        'hard_x_lims': hard_x_lims,
                                                        "k_best": k_best})
            return result

        print(f"data's range: [{arr.min()}, {arr.max()}]")
        min_x, max_x = np.percentile(arr, hard_x_lims[0] * 100), np.percentile(arr, hard_x_lims[1] * 100)
        kde_f = gaussian_kde(arr)
        x = np.linspace(arr.min(), arr.max(), 10000)
        y = kde_f(x)
        if return_heuristic:
            if k_best > 1:
                print("Only return a group of params because the return_heuristic is set to True")
            c1, c2 = self.get_init_for_bimodal(x, y, max_x=max_x, min_x=min_x)
            c1, c2 = min(c1, c2), max(c1, c2)
            params_arr = np.array([[1, c2, 1, 1, c1, 1]])
        else:
            params_arr, c1, c2 = self._bimodal_fit(x, y, max_x=max_x, min_x=min_x, k=k_best)

        result = rFASTCORMICSThresholdAnalysis(log={"use_first_guess": return_heuristic,
                                                   "cut_off": cut_off,
                                                   'hard_x_lims': hard_x_lims,
                                                   "k_best": k_best})
        right_c = None if return_heuristic else np.array([self.gaussian_dist(x, *tuple(params_arr[k, 0:3])) for k in range(k_best)])
        left_c = None if return_heuristic else np.array([self.gaussian_dist(x, *tuple(params_arr[k, 3:6])) for k in range(k_best)])
        result.add_result(dict(x=x,
                               y=y,
                               exp_th_arr=params_arr[:, 1],
                               nonexp_th_arr=params_arr[:, 4],
                               right_curve_arr=right_c,
                               left_curve_arr=left_c,
                               init_exp=c2,
                               init_nonexp=c1))

        return result


class RankBased(ThresholdFinder):
    """Base class for threshold finders based on data ranking (e.g., percentiles)."""
    def __init__(self):
        """Initializes the RankBased finder."""
        super().__init__()

    def find_threshold(self, **kwargs):
        """
        Abstract method for rank-based threshold finding.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError()


class PercentileThreshold(RankBased):
    """Finds thresholds based on simple percentiles of the data."""
    def __init__(self):
        """Initializes the PercentileThreshold finder."""
        super().__init__()

    @timing
    def find_threshold(self,
                       data: Union[np.ndarray, pd.Series, dict, list],
                       p: Union[int, float, List[int], List[float], np.ndarray],
                       exp_p: Optional[Union[int, float]] = None,
                       non_exp_p: Optional[Union[int, float]] = None,
                       **kwargs) -> PercentileThresholdAnalysis:
        """
        Calculate thresholds based on specified percentiles of the input data.

        Parameters
        ----------
        data
        p: float, int, or a sequence of int or float
            Percentile or sequence of percentiles to compute, which must be between 0 and 100 inclusive.
        exp_p: float or int
            Percentile to compute the expression threshold.
            If None, the maximum value of the percentiles is returned as the exp_th.
        non_exp_p: float or int
            Percentile to compute the non-expression threshold.
            If None, the minimum value of the percentiles is returned as the non_exp_th.
        kwargs: dict
            kwargs used by np.percentile().

        Returns
        -------
        result: PercentileThresholdAnalysis

        """
        if isinstance(p, float) or isinstance(p, int):
            if not (0 <= p <= 100):
                raise ValueError(f"p should be a number between 0 and 100. Got {p} instead")
            if exp_p is not None and exp_p != p:
                raise ValueError(f"exp_p ({exp_p}) must be equal to p ({p}) if p is a value")
            if non_exp_p is not None and non_exp_p != p:
                raise ValueError(f"non_exp_p ({non_exp_p}) must be equal to p ({p}) if p is a value")
        else:
            if not all([0 <= pi <= 100 for pi in p]):
                raise ValueError(f"Every number in p should be between 0 and 100. Got {p} instead")
            if exp_p is not None and exp_p not in p:
                raise ValueError(f"exp_p ({exp_p}) must be in the p ({p}) if p is a sequence")
            if non_exp_p is not None and non_exp_p not in p:
                raise ValueError(f"non_exp_p ({non_exp_p}) must be in the p ({p}) if p is a sequence")
            if exp_p is not None and non_exp_p is not None and exp_p < non_exp_p:
                raise ValueError("exp_p must be greater or equal to non_exp_p.")

        if isinstance(data, pd.Series):
            arr = data.values
        elif isinstance(data, dict):
            arr = np.array(list(data.values()))
        else:
            arr = data
        arr = arr[np.isfinite(arr)]
        result = PercentileThresholdAnalysis(log={"p": p, "exp_p": exp_p, "non_exp_p": non_exp_p})
        if isinstance(p, list) or isinstance(p, np.ndarray):
            ths = [np.percentile(arr, q=pi, **kwargs) for pi in p]
            ths = pd.Series(ths, index=[f"p={pi}"for pi in p])
            exp_th = ths.values.max() if exp_p is None else ths[exp_p]
            non_exp_th = ths.values.min() if non_exp_p is None else ths[non_exp_p]
        else:
            exp_th = np.percentile(arr, q=p, **kwargs)
            non_exp_th = exp_th
            ths = None

        result.add_result(dict(data=arr,
                               exp_th=exp_th,
                               non_exp_th=non_exp_th,
                               threshold_series=ths))
        return result


class LocalThreshold(RankBased):
    """
    Finds local expression thresholds based on percentiles within groups.

    Calculates gene-specific thresholds for different sample groups based on
    a specified percentile `p`. Optionally applies global 'on'/'off' thresholds
    to override local thresholds for genes consistently high or low across a group.
    """
    def __init__(self):
        """Initializes the LocalThreshold finder."""
        super().__init__()

    @timing
    def find_threshold(self,
                       data: Union[np.ndarray, pd.DataFrame],
                       p: Union[int, float], # Simplified: expects single percentile for local
                       global_on_p : Optional[Union[int, float]] = 90,
                       global_off_p: Optional[Union[int, float]] = 10,
                       global_on_th: Optional[Union[int, float]] = None,
                       global_off_th: Optional[Union[int, float]] = None,
                       groups: Optional[Union[pd.Series, dict]] = None,
                       genes: Optional[List[str]] = None, # Added for array input
                       samples: Optional[List[str]] = None, # Added for array input
                       **kwargs) -> LocalThresholdAnalysis:
        """
        Calculate local, gene-specific expression thresholds for sample groups.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            Input expression data (genes x samples). If ndarray, `genes` and
            `samples` arguments must be provided.
        p : Union[int, float]
            The percentile (0-100) used to calculate the local threshold for each gene
            within each group.
        global_on_p : Optional[Union[int, float]], optional
            Percentile (0-100) to determine the global 'on' threshold. If a gene's
            minimum expression within a group exceeds this percentile's value
            across the group, the global threshold overrides the local one.
            Used if `global_on_th` is None. Default is 90.
        global_off_p : Optional[Union[int, float]], optional
            Percentile (0-100) to determine the global 'off' threshold. If a gene's
            maximum expression within a group is below this percentile's value
            across the group, the global threshold overrides the local one.
            Used if `global_off_th` is None. Default is 10.
        global_on_th : Optional[Union[int, float]], optional
            Explicit value for the global 'on' threshold. Overrides `global_on_p`.
            Default is None.
        global_off_th : Optional[Union[int, float]], optional
            Explicit value for the global 'off' threshold. Overrides `global_off_p`.
            Default is None.
        groups : Optional[Union[pd.Series, dict]], optional
            Mapping samples to groups. If dict, keys are group names, values are
            lists of sample names. If Series, index is sample names, values are
            group names. If None, all samples are treated as one group named 'exp_th'.
            Default is None.
        genes : Optional[List[str]], optional
            List of gene names corresponding to rows if `data` is an ndarray.
            Required if `data` is an ndarray. Default is None.
        samples : Optional[List[str]], optional
            List of sample names corresponding to columns if `data` is an ndarray.
            Required if `data` is an ndarray. Default is None.
        **kwargs
            Additional keyword arguments (currently unused).

        Returns
        -------
        local_threshold_analysis: LocalThresholdAnalysis
            The result object contains a local threshold dataframe.
            The dataframe will be N_gene x N_group(if the groups arg is specified) containing the expression thresholds

        """
        assert 0 <= p <= 100
        if isinstance(data, pd.DataFrame):
            arr = data.values
            genes = data.index
            samples = data.columns
        else:
            arr = data
            genes = kwargs.get("genes")
            samples = kwargs.get("samples")
        if isinstance(groups, dict):
            if all([isinstance(v, list) for v in groups.values()]):
                groups = pd.Series({vi: k for k, v in groups.items() for vi in v})
            else:
                groups = pd.Series(groups)
        elif groups is None:
            groups = pd.Series(["exp_th" for _ in range(len(samples))], index=samples)
        group_list = groups.unique()

        arr = np.nan_to_num(arr, posinf=np.nan, neginf=np.nan)
        exp_ths = pd.DataFrame({grp: np.nanpercentile(arr[:, samples.isin(groups[groups == grp].index)], q=p, axis=1)
                                for grp in group_list}, index=genes)
        global_on_dic, global_off_dic = {}, {}

        if global_on_p is not None or global_on_th is not None:
            for grp in group_list:
                glob_on_th_grp = np.nanpercentile(arr[:, samples.isin(groups[groups == grp].index)], q=global_on_p) \
                    if global_on_th is None else global_on_th
                global_on_dic[grp] = glob_on_th_grp[grp] if isinstance(glob_on_th_grp, dict) else glob_on_th_grp
                exp_ths.loc[exp_ths.index[np.nanmin(arr[:, samples.isin(groups[groups == grp].index)],
                                                    axis=1) > global_on_dic[grp]], grp] = glob_on_th_grp
        if global_off_p is not None or global_off_th is not None:
            for grp in group_list:
                glob_off_th_grp = np.nanpercentile(arr[:, samples.isin(groups[groups == grp].index)], q=global_off_p) \
                    if global_off_th is None else global_off_th
                global_off_dic[grp] = glob_off_th_grp[grp] if isinstance(glob_off_th_grp, dict) else glob_off_th_grp
                exp_ths.loc[exp_ths.index[np.nanmax(arr[:, samples.isin(groups[groups == grp].index)],
                                                    axis=1) < global_off_dic[grp]], grp] = glob_off_th_grp

        result = LocalThresholdAnalysis(log={"p": p})
        result.add_result(dict(exp_ths=exp_ths,
                               global_on_th=pd.Series(global_on_dic),
                               global_off_th=pd.Series(global_off_dic),
                               data=pd.DataFrame(data=arr, index=genes, columns=samples),
                               groups=groups))
        return result


threshold_finders = ThresholdFinders()
threshold_finders.register("rFASTCORMICS", rFASTCORMICSThreshold)
threshold_finders.register("percentile", PercentileThreshold)  # which is also called global threshold in some papers
threshold_finders.register("local", LocalThreshold)  # LocalT1 and LocalT2


ALL_THRESHOLD_METHODS = Union[rFASTCORMICSThreshold, PercentileThreshold, LocalThreshold]
