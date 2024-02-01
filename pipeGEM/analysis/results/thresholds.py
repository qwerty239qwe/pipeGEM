from typing import Union
import pandas as pd

from ._base import *
from pipeGEM.plotting import rFastCormicThresholdPlotter, PercentileThresholdPlotter, LocalThresholdPlotter


class rFASTCORMICSThresholdAnalysis(BaseAnalysis):
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