import json

from ._base import *
from pipeGEM.plotting import rFastCormicThresholdPlotter, PercentileThresholdPlotter


class rFastCormicThresholdAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log=log)
        self._data = ()
        self._exp_th = None
        self._nonexp_th = None
        self._right_curve = None
        self._left_curve = None

    def save(self, file_path):
        result_dic = {"exp_th": self._exp_th,
                      "non_exp_th": self._nonexp_th}

        with open(file_path, "w") as f:
            json.dump(result_dic, f)

    def add_result(self, x, y, exp_th, nonexp_th, right_curve, left_curve):
        self._data = (x, y)
        self._exp_th = exp_th
        self._nonexp_th = nonexp_th
        self._right_curve = right_curve
        self._left_curve = left_curve

    @property
    def exp_th(self):
        return self._exp_th

    @property
    def non_exp_th(self):
        return self._nonexp_th

    def plot(self,
             dpi=150,
             prefix="",
             *args,
             **kwargs):
        pltr = rFastCormicThresholdPlotter(dpi=dpi, prefix=prefix)
        pltr.plot(x=self._data[0],
                  y=self._data[1],
                  exp_th=self._exp_th,
                  nonexp_th=self._nonexp_th,
                  right_curve=self._right_curve,
                  left_curve=self._left_curve,
                  *args,
                  **kwargs)


class PercentileThresholdAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log=log)
        self._data = ()
        self._exp_th = None

    def add_result(self, data, exp_th):
        self._data = data
        self._exp_th = exp_th

    @property
    def exp_th(self):
        return self._exp_th

    def plot(self,
             dpi=150,
             prefix="",
             *args,
             **kwargs):
        pltr = PercentileThresholdPlotter(dpi=dpi, prefix=prefix)
        pltr.plot(data=self._data,
                  exp_th=self._exp_th,
                  *args,
                  **kwargs)


class LocalThresholdAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log=log)
