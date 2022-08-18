from pipeGEM.plotting import FBAPlotter, FVAPlotter, SamplingPlotter, \
    rFastCormicThresholdPlotter, PercentileThresholdPlotter, ComponentComparisonPlotter
from functools import wraps
from time import time

import json
import pandas as pd


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        if isinstance(result, list) or isinstance(result, tuple):
            result[0].add_running_time(te - ts)
        else:
            result.add_running_time(te - ts)
        return result
    return wrap


class BaseAnalysis:
    def __init__(self, log):
        self._log = log  # analysis record (parameter, model, and data name)

    def add_running_time(self, t):
        self._log["running_time"] = t

    @property
    def log(self):
        return self._log

    def save(self, file_path):
        pass

    @classmethod
    def load(cls, file_name, **kwargs):
        pass

    def plot(self, **kwargs):
        raise NotImplementedError()


class FluxAnalysis(BaseAnalysis):
    def __init__(self, log):
        super(FluxAnalysis, self).__init__(log)
        self._sol = None
        self._df = None

    @classmethod
    def aggregate(cls, analyses, method, log, **kwargs):
        new = cls(log=log)
        if method == "concat":
            dfs = []
            for a in analyses:
                one_df = a.result.reset_index().rename(columns={"index": "Reaction"})
                one_df["name"] = a.log["name"]
                dfs.append(one_df)
            new._df = pd.concat(dfs, axis=0)
            new._df["name"] = pd.Categorical(new._df["name"])
        else:
            dfs = []
            for a in analyses:
                one_df = a.result["fluxes"]
                one_df.name = f"fluxes_{a.log['name']}"
                dfs.append(one_df)
            new._df = pd.concat(dfs, axis=1)
            new._df = getattr(new._df, method)(axis=1).to_frame()
            new._df.columns = ["fluxes"]
        return new

    def add_result(self, sol):
        raise NotImplementedError()

    def plot(self, **kwargs):
        raise NotImplementedError()


class FBA_Analysis(FluxAnalysis):
    def __init__(self, log):
        super().__init__(log)

    @property
    def result(self):
        return self._df

    @property
    def solution(self):
        return self._sol

    def add_result(self, result):
        self._sol = result
        self._df = self._sol.to_frame()

    def plot(self,
             dpi=150,
             prefix="FBA_",
             *args,
             **kwargs):
        pltr = FBAPlotter(dpi, prefix)
        pltr.plot(flux_df=self._df,
                  *args,
                  **kwargs)


class FVA_Analysis(FluxAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._df = None

    @classmethod
    def aggregate(cls, analyses, method, log, **kwargs):
        new = cls(log=log)
        if method == "concat":
            dfs = []
            for a in analyses:
                one_df = a.result.reset_index().rename(columns={"index": "Reaction"})
                one_df["name"] = a.log["name"]
                dfs.append(one_df)
            new._df = pd.concat(dfs, axis=0)
            new._df["name"] = pd.Categorical(new._df["name"])
        else:
            min_dfs, max_dfs = [], []
            for a in analyses:
                min_df, max_df = a.result["minimum"], a.result["maximum"]
                min_df.name, max_df.name = f"minimum_{a.log['name']}", f"maximum_{a.log['name']}"
                min_dfs.append(min_df)
                max_dfs.append(max_df)
            new_min_df, new_max_df = pd.concat(min_dfs, axis=1), pd.concat(max_dfs, axis=1)
            new_min_df, new_max_df = getattr(new_min_df, method)(axis=1).to_frame(), getattr(new_max_df, method)(axis=1).to_frame()
            new._df = {"minimum": new_min_df, "maximum": new_max_df}
        return new

    def add_result(self, result):
        self._df = result

    @property
    def result(self):
        return self._df

    def plot(self,
             rxn_ids,
             dpi=150,
             prefix="FVA_",
             *args,
             **kwargs):
        pltr = FVAPlotter(dpi=dpi, prefix=prefix)
        pltr.plot(rxn_ids=rxn_ids,
                  flux_df=self._df,
                  *args,
                  **kwargs)


class SamplingAnalysis(FluxAnalysis):
    def __init__(self, log):
        super(SamplingAnalysis, self).__init__(log)
        self._df_dic = {}

    @classmethod
    def aggregate(cls, analyses, method, log, **kwargs):
        new = cls(log=log)
        if method == "concat":
            for a in analyses:
                new._df_dic.update(a._df_dic)
        return new

    @property
    def result(self):
        return self._df_dic

    def add_result(self, result):
        self._df_dic = result.melt(var_name="rxn_id", value_name="flux")

    def plot(self,
             dpi=150,
             prefix="sampling_",
             *args,
             **kwargs):
        pltr = SamplingPlotter(dpi=dpi, prefix=prefix)
        pltr.plot(flux_df_dic=self._df_dic,
                  *args,
                  **kwargs)


class FastCCAnalysis(BaseAnalysis):
    def __init__(self, log):
        self._consist_model = None
        self._removed_rxn_ids = None
        super(FastCCAnalysis, self).__init__(log)

    def add_result(self, result):
        self._consist_model = result.get("model")
        self._removed_rxn_ids = result.get("removed_rxn_ids")

    @property
    def consist_model(self):
        return self._consist_model

    @property
    def removed_rxn_ids(self):
        return self._removed_rxn_ids


class TaskAnalysis(BaseAnalysis):
    result_df_default_colnames = ['Passed', 'Should fail',
                                  'Missing mets', 'Status',
                                  'Obj_value', 'Obj_rxns',
                                  'Sink Status']

    def __init__(self, log):
        self._result_df = None
        self._model_score = 0
        self._task_support_rxns = {}
        self._task_support_rxn_fluxes = {}

        self._rxn_supps = {}
        super(TaskAnalysis, self).__init__(log)

    @classmethod
    def load(cls, file_name, **kwargs):

        with open(file_name) as json_file:
            data = json.load(json_file)
            task_analysis = cls(log=data.pop("log"))
            result_dic = {k: {task_id: task_results[k] for task_id, task_results in data.items()}
                          for k in cls.result_df_default_colnames}

            for task_id, task_results in data.items():
                tr, trf, rs = task_results.get("task_support_rxns"), \
                              task_results.get("task_support_rxn_fluxes"), \
                              task_results.get("rxn_supps")
                if tr is not None:
                    task_analysis._task_support_rxns[task_id] = tr
                if trf is not None:
                    task_analysis._task_support_rxn_fluxes[task_id] = trf
                if rs is not None:
                    task_analysis._rxn_supps[task_id] = rs
            result_dic.update({"task_support_rxns": task_analysis._task_support_rxns,
                               "task_support_rxn_fluxes": task_analysis._task_support_rxn_fluxes,
                               "rxn_supps": task_analysis._rxn_supps})

            task_analysis._result_df = pd.DataFrame(result_dic)

        return task_analysis

    def save(self, file_name):
        result_dic = {}
        for c in self.result_df_default_colnames:
            for k, v in self.result_df[c].to_dict().items():
                if k not in result_dic:
                    result_dic[k] = {}
                if c != "Obj_rxns":
                    result_dic[k][c] = v if not pd.isna(v) else None
                else:
                    result_dic[k][c] = [vi.id for vi in v]

        for attr_name, attr in zip(["task_support_rxns", "task_support_rxn_fluxes", "rxn_supps"],
                                   [self._task_support_rxns, self._task_support_rxn_fluxes, self._rxn_supps]):
            for k, v in attr.items():
                if not(isinstance(v, list) or isinstance(v, dict)) and pd.isna(v):
                    result_dic[k][attr_name] = None
                else:
                    result_dic[k][attr_name] = v
        result_dic["log"] = self._log
        with open(file_name, "w") as f:
            json.dump(result_dic, f)

    @property
    def model_score(self):
        return self._model_score

    @property
    def task_support_rxns(self):
        return self._task_support_rxns

    @property
    def result_df(self):
        return self._result_df

    def add_result(self, result_df, score):
        self._result_df = result_df
        self._model_score = score
        self._task_support_rxns = dict(self._result_df["task_support_rxns"].iteritems())
        self._task_support_rxn_fluxes = dict(self._result_df["task_support_rxn_fluxes"].iteritems())
        self._rxn_supps = dict(self._result_df["rxn_supps"].iteritems())

    def get_task_support_rxns(self, task_id, include_supps=True):
        return self._task_support_rxns[task_id] + (self._rxn_supps[task_id]
                                                   if include_supps and task_id in self._rxn_supps and isinstance(self._rxn_supps[task_id], list)
                                                   else [])


class EFluxAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._rxn_bounds = None
        self._rxn_scores = None

    @property
    def rxn_bounds(self):
        return self._rxn_bounds

    def add_result(self, rxn_bounds, rxn_scores):
        self._rxn_bounds = rxn_bounds
        self._rxn_scores = rxn_scores


class GIMMEAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._rxn_coefficents = None

    @property
    def rxn_coefficents(self):
        return self._rxn_coefficents

    def add_result(self, rxn_coefficents, rxn_scores):
        self._rxn_coefficents = rxn_coefficents
        self._rxn_scores =rxn_scores


class RIPTiDePruningAnalysis(BaseAnalysis):
    def  __init__(self, log):
        super().__init__(log)
        self._model = None
        self._removed_rxns = None
        self._obj_dict = None

    @property
    def model(self):
        return self._model

    def add_result(self, model, removed_rxns, obj_dict):
        self._model = model
        self._removed_rxns = removed_rxns
        self._obj_dict = obj_dict


class RIPTiDeSamplingAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._model = None
        self._sampling_result = None

    def add_result(self, sampling_result):
        self._sampling_result = sampling_result


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


class rFastCormicAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._model = None
        self._rxn_ids = None
        self._removed_rxn_ids = None
        self._core_rxns = None
        self._noncore_rxns = None
        self._nonP_rxns = None
        self._threshold_analysis = None

    @property
    def threshold_analysis(self):
        return self._threshold_analysis

    @property
    def result_model(self):
        return self._model

    def add_result(self, fastcore_result, core_rxns, noncore_rxns, nonP_rxns, threshold_analysis):
        self._model = fastcore_result.get("model")
        self._rxn_ids = fastcore_result.get("rxn_ids")
        self._removed_rxn_ids = fastcore_result.get("removed_rxn_ids")
        self._core_rxns = core_rxns
        self._noncore_rxns = noncore_rxns
        self._nonP_rxns = nonP_rxns
        self._threshold_analysis = threshold_analysis


class ComparisonAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)


class ComponentComparisonAnalysis(ComparisonAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._result = None

    def add_result(self, result):
        self._result = result

    def plot(self,
             dpi=150,
             prefix="",
             *args,
             **kwargs):
        pltr = ComponentComparisonPlotter(dpi=dpi, prefix=prefix)
        pltr.plot(result=self._result,
                  *args,
                  **kwargs)


def combine(analyses, method, log, **kwargs):
    if len(analyses) < 2:
        raise ValueError("Analyses should be a container with more than 2 analysis objects")
    if isinstance(analyses[0], FluxAnalysis):
        return analyses[0].__class__.aggregate(analyses, method, log, **kwargs)

    raise ValueError("These analysis objects have no combining function.")