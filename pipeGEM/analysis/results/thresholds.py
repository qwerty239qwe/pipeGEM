import json

from ._base import *
from pipeGEM.plotting import rFastCormicThresholdPlotter, PercentileThresholdPlotter, LocalThresholdPlotter


class rFASTCORMICSThresholdAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log=log)

    @property
    def exp_th(self):
        return self._result["exp_th_arr"][0]

    @property
    def non_exp_th(self):
        return self._result["nonexp_th_arr"][0]

    @property
    def init_threshold(self):
        return self._result["init_exp"], self._result["init_nonexp"]

    def get_other_exp_th(self, k):
        return self._result["exp_th_arr"][k]

    def get_other_non_exp_th(self, k):
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

    @property
    def exp_th(self):
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
    def local_ths(self):
        return self._result["local_ths"]

    @property
    def global_off_th(self):
        return self._result["global_off_th"]

    @property
    def global_on_th(self):
        return self._result["global_on_th"]

    def _get_group_dic(self):
        groups = self._groups.unique()
        return {g: self._result["groups"][self._result["groups"] == g].to_list()
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
                  groups=groups,
                  local_th=self._result["local_ths"],
                  global_on_th=self._result["global_on_th"],
                  global_off_th=self._result["global_off_th"],
                  group_dict=self._get_group_dic(),
                  )