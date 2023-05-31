import json

from ._base import *
from pipeGEM.plotting import rFastCormicThresholdPlotter, PercentileThresholdPlotter, LocalThresholdPlotter


class rFastCormicThresholdAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log=log)
        self._data = ()
        self._exp_th_arr = None
        self._nonexp_th_arr = None
        self._right_curve_arr = None
        self._left_curve_arr = None
        self._init_exp = None
        self._init_nonexp = None

    def save(self, file_path, index=0):
        result_dic = {"exp_th": self._exp_th_arr[index],
                      "non_exp_th": self._nonexp_th_arr[index]}

        with open(file_path, "w") as f:
            json.dump(result_dic, f)

    def add_result(self, x, y, exp_th, nonexp_th, right_curve, left_curve, init_exp, init_nonexp):
        self._data = (x, y)
        self._exp_th_arr = exp_th
        self._nonexp_th_arr = nonexp_th
        self._right_curve_arr = right_curve
        self._left_curve_arr = left_curve
        self._init_exp = init_exp
        self._init_nonexp = init_nonexp

    @property
    def exp_th(self):
        return self._exp_th_arr[0]

    @property
    def non_exp_th(self):
        return self._nonexp_th_arr[0]

    @property
    def init_threshold(self):
        return self._init_exp, self._init_nonexp

    def get_other_exp_th(self, k):
        return self._exp_th_arr[k]

    def get_other_non_exp_th(self, k):
        return self._nonexp_th_arr[k]

    def plot(self,
             dpi=150,
             prefix="",
             k=0,
             *args,
             **kwargs):
        pltr = rFastCormicThresholdPlotter(dpi=dpi, prefix=prefix)
        pltr.plot(x=self._data[0],
                  y=self._data[1],
                  exp_th=self._exp_th_arr[k],
                  nonexp_th=self._nonexp_th_arr[k],
                  right_curve=self._right_curve_arr[k] if self._right_curve_arr is not None else None,
                  left_curve=self._left_curve_arr[k] if self._left_curve_arr is not None else None,
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
        self._groups = log.get("groups")
        self._local_ths = None
        self._data = None
        self._global_off_th = None
        self._global_on_th = None

    def save(self, file_path):
        pass

    def add_result(self,
                   data,
                   local_ths,
                   global_on_ths,
                   global_off_ths):
        self._local_ths = local_ths
        self._global_off_th = global_on_ths
        self._global_on_th = global_off_ths
        self._data = data

    @property
    def local_ths(self):
        return self._local_ths

    @property
    def global_off_th(self):
        return self._global_off_th

    @property
    def global_on_th(self):
        return self._global_on_th

    def _get_group_dic(self):
        groups = self._groups.unique()
        return {g: self._groups[self._groups == g].to_list() for g in groups}

    def plot(self,
             genes,
             groups="all",
             dpi=150,
             prefix="",
             *args,
             **kwargs):
        pltr = LocalThresholdPlotter(dpi=dpi, prefix=prefix)
        pltr.plot(data=self._data,
                  groups=groups,
                  local_th=self._local_ths,
                  global_on_th=self._global_on_th,
                  global_off_th=self._global_off_th,
                  group_dict=self._get_group_dic(),
                  )