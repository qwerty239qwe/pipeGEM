from ._base import *


class ModelScalingResult(BaseAnalysis):
    def __init__(self, log):
        super(ModelScalingResult, self).__init__(log=log)
        self._result_saving_params["diff_A"] = {"fm_name": "NDArrayFloat"}
        self._result_saving_params["rescaled_A"] = {"fm_name": "NDArrayFloat"}