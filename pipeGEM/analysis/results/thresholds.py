from typing import Union
import pandas as pd

from ._base import *
from pipeGEM.plotting import rFastCormicThresholdPlotter, PercentileThresholdPlotter, LocalThresholdPlotter


class rFASTCORMICSThresholdAnalysis(BaseAnalysis):
    """
    Analysis result object for thresholds found using the rFASTCORMICS method.

    Stores the results of fitting a bimodal Gaussian distribution to the expression
    data's Kernel Density Estimate (KDE). Provides access to the calculated
    expression and non-expression thresholds, the fitted curves, and the
    original KDE data. Also includes plotting functionality.

    Attributes
    ----------
    exp_th : float
        The primary expression threshold (mean of the higher-expression Gaussian).
    non_exp_th : float
        The primary non-expression threshold (mean of the lower-expression Gaussian).
    init_threshold : tuple[float, float]
        The initial heuristic guesses for the expression and non-expression thresholds.
    _result : dict
        Dictionary holding the detailed results:
        - "x": np.ndarray, x-values for the KDE.
        - "y": np.ndarray, y-values (density) for the KDE.
        - "exp_th_arr": np.ndarray, array of best expression thresholds found (ranked).
        - "nonexp_th_arr": np.ndarray, array of best non-expression thresholds found (ranked).
        - "right_curve_arr": np.ndarray | None, array of y-values for the fitted higher-expression Gaussian curves.
        - "left_curve_arr": np.ndarray | None, array of y-values for the fitted lower-expression Gaussian curves.
        - "init_exp": float, initial guess for expression threshold.
        - "init_nonexp": float, initial guess for non-expression threshold.
    """
    def __init__(self, log):
        super().__init__(log=log)
        for attr in ["x", "y", "exp_th_arr",
                     "nonexp_th_arr", "right_curve_arr", "left_curve_arr"]:
            self._result_saving_params[attr] = {"fm_name": "NDArrayFloat"}

    @property
    def exp_th(self) -> float:
        return self._result["exp_th_arr"][0]

    @property
    def non_exp_th(self) -> float:
        return self._result["nonexp_th_arr"][0]

    @property
    def init_threshold(self) -> (float, float):
        return self._result["init_exp"], self._result["init_nonexp"]

    def get_other_exp_th(self, k) -> float:
        return self._result["exp_th_arr"][k]

    def get_other_non_exp_th(self, k) -> float:
        return self._result["nonexp_th_arr"][k]

    def plot(self,
             dpi=150,
             prefix="",
             k=0,
             *args,
             **kwargs):
        pltr = rFastCormicThresholdPlotter(dpi=dpi, prefix=prefix)
        pltr.plot(x=self._result["x"],
                  y=self._result["y"],
                  exp_th=self._result["exp_th_arr"][k],
                  nonexp_th=self._result["nonexp_th_arr"][k],
                  right_curve=self._result["right_curve_arr"][k] if self._result["right_curve_arr"] is not None else None,
                  left_curve=self._result["left_curve_arr"][k] if self._result["left_curve_arr"] is not None else None,
                  *args,
                  **kwargs)


class PercentileThresholdAnalysis(BaseAnalysis):
    """
    Analysis result object for thresholds found using simple percentiles.

    Stores the results of calculating thresholds based on specified percentiles
    of the expression data. Provides access to the calculated expression threshold
    (which might be a single value or a series if multiple percentiles were used)
    and the original data. Includes plotting functionality.

    Attributes
    ----------
    exp_th : float | pd.Series
        The calculated expression threshold(s). If a single percentile `p` was used,
        this is a float. If multiple percentiles were specified, this might be the
        highest threshold (default) or a specific one if `exp_p` was set.
        Refer to the `threshold_series` for all calculated percentile values.
    non_exp_th : float | pd.Series
        The calculated non-expression threshold(s). Similar logic to `exp_th`,
        using the lowest threshold by default or a specific one if `non_exp_p` was set.
    _result : dict
        Dictionary holding the detailed results:
        - "data": np.ndarray, the filtered input expression data used for calculation.
        - "exp_th": float, the final expression threshold selected.
        - "non_exp_th": float, the final non-expression threshold selected.
        - "threshold_series": pd.Series | None, Series containing thresholds for all
          calculated percentiles (index: "p=value"), or None if only one `p` was given.
    """
    def __init__(self, log):
        super().__init__(log=log)
        self._result_saving_params["data"] = {"fm_name": "NDArrayFloat"}

    @property
    def exp_th(self) -> Union[float, pd.Series]:
        return self._result["exp_th"]

    def plot(self,
             dpi=150,
             prefix="",
             *args,
             **kwargs):
        pltr = PercentileThresholdPlotter(dpi=dpi, prefix=prefix)
        pltr.plot(data=self._result["data"],
                  exp_th=self._result["exp_th"],
                  *args,
                  **kwargs)


class LocalThresholdAnalysis(BaseAnalysis):
    """
    Analysis result object for locally calculated expression thresholds.

    Stores gene-specific expression thresholds calculated for different sample groups
    based on within-group percentiles. Also stores optional global 'on' and 'off'
    thresholds used to override local thresholds for consistently high/low genes.
    Includes plotting functionality for visualizing expression distributions and thresholds.

    Attributes
    ----------
    exp_ths : pd.DataFrame
        DataFrame containing the local expression thresholds (genes x groups).
    global_off_th : pd.Series
        Series containing the global 'off' threshold for each group (index: group name).
        Genes with maximum expression below this in a group use this threshold.
    global_on_th : pd.Series
        Series containing the global 'on' threshold for each group (index: group name).
        Genes with minimum expression above this in a group use this threshold.
    _result : dict
        Dictionary holding the detailed results:
        - "exp_ths": pd.DataFrame, the local thresholds (genes x groups).
        - "global_on_th": pd.Series, global 'on' thresholds per group.
        - "global_off_th": pd.Series, global 'off' thresholds per group.
        - "data": pd.DataFrame, the input expression data (genes x samples).
        - "groups": pd.Series, mapping of samples (index) to group names (values).
    """
    def __init__(self, log):
        super().__init__(log=log)

    @property
    def exp_ths(self) -> pd.Series:
        return self._result["exp_ths"]

    @property
    def global_off_th(self) -> pd.Series:
        return self._result["global_off_th"]

    @property
    def global_on_th(self) -> pd.Series:
        return self._result["global_on_th"]

    def _get_group_dic(self):
        groups = self._result["groups"].unique()
        return {g: self._result["groups"][self._result["groups"] == g].index.to_list()
                for g in groups}

    def plot(self,
             genes,
             groups="all",
             dpi=150,
             prefix="",
             *args,
             **kwargs):
        pltr = LocalThresholdPlotter(dpi=dpi, prefix=prefix)
        pltr.plot(data=self._result["data"],
                  genes=genes,
                  groups=groups,
                  local_th=self._result["exp_ths"],
                  global_on_th=self._result["global_on_th"],
                  global_off_th=self._result["global_off_th"],
                  group_dic=self._get_group_dic(),
                  )


ALL_THRESHOLD_ANALYSES = Union[rFASTCORMICSThresholdAnalysis, PercentileThresholdAnalysis, LocalThresholdAnalysis, ]
