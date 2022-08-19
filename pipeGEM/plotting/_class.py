import string
from pathlib import Path
from typing import Union, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from ._flux import plot_fba, plot_fva, plot_sampling
from .curve import plot_rFastCormic_thresholds, plot_percentile_thresholds
from .heatmap import plot_heatmap
from .scatter import plot_PCA, plot_embedding


class BasePlotter:
    def __init__(self, dpi=150, prefix=""):
        self.dpi = dpi
        self.prefix = prefix

    @staticmethod
    def format_file_name(name_format, plotting_kws):
        if "name_format" in plotting_kws:
            name_format = plotting_kws.pop("name_format")

        sf = string.Formatter()
        format_kws = [x[1] for x in sf.parse(name_format) if x[1] is not None]
        if all([f in plotting_kws for f in format_kws]):
            naming_kws = {f: plotting_kws.pop(f) for f in format_kws}
            return name_format.format(**naming_kws) if all([v is not None for v in naming_kws.values()]) else None
        return None

    @staticmethod
    def extract_kws(kws: dict, keys, default_kws):
        extracted = {}
        for k in keys:
            default = default_kws.get(k)
            extracted[k] = kws.pop(k, default)
        return extracted

    @staticmethod
    def _save_fig(file_name, prefix, dpi=150, g=None):
        if prefix is None:
            prefix = ""

        if isinstance(file_name, str) and "/" in file_name:
            file_path = Path("/".join(file_name.split("/")[:-1]))
            file_name = Path(prefix + file_name.split("/")[-1])
        else:
            file_name, file_path = Path(prefix + str(file_name)), Path("./")

        if g is None:
            plt.savefig(file_path / file_name, dpi=dpi, bbox_inches='tight')
        else:
            g.savefig(file_path / file_name, dpi=dpi, bbox_inches='tight')

    def plot(self, *args, **kwargs):
        """
        This is a wrapper function for dealing with styling, file saving, and some other stuffs

        Returns
        -------

        """

        plotting_kws = self.extract_kws(kwargs,
                                        ["file_name", "prefix", "dpi"],
                                        default_kws={"dpi": self.dpi, "prefix": self.prefix})
        self.add_style()
        info_kws = self.plot_func(*args, **kwargs)
        name_format = info_kws.pop("name_format") if "name_format" in info_kws else "{file_name}"
        if info_kws is not None:
            plotting_kws.update(info_kws)
            updated_name = self.format_file_name(name_format, plotting_kws)
            plotting_kws["file_name"] = updated_name
            if plotting_kws["file_name"] is not None:
                print("saving ", plotting_kws["file_name"])
                self._save_fig(**plotting_kws)
            plt.show()
        return info_kws

    def plot_func(self, *args, **kwargs):
        raise NotImplementedError("This class should be inherited")

    def add_style(self):
        plt.style.use("seaborn")


class FBAPlotter(BasePlotter):
    def __init__(self, dpi=150, prefix="FBA_"):
        super(FBAPlotter, self).__init__(dpi, prefix)

    def plot_func(self,
                  flux_df: pd.DataFrame,
                  rxn_ids: Union[List[str], Dict[str, str]],
                  **kwargs
                  ):
        return plot_fba(flux_df=flux_df,
                        rxn_ids=rxn_ids,
                        **kwargs)


class pFBAPlotter(FBAPlotter):
    def __init__(self, dpi=150, prefix="pFBA_"):
        super(pFBAPlotter, self).__init__(dpi, prefix)


class FVAPlotter(BasePlotter):
    def __init__(self, dpi=150, prefix="FVA_"):
        super(FVAPlotter, self).__init__(dpi, prefix)

    def plot_func(self,
                  fva_df: pd.DataFrame,
                  rxn_ids: Union[List[str], Dict[str, str]],
                  **kwargs
                  ):
        return plot_fva(fva_df=fva_df,
                        rxn_ids=rxn_ids,
                        **kwargs)


class SamplingPlotter(BasePlotter):
    def __init__(self, dpi=150, prefix="SMP_"):
        super(SamplingPlotter, self).__init__(dpi, prefix)

    def plot_func(self,
                  flux_df_dic: Dict[str, pd.DataFrame],
                  rxn_ids: Union[List[str], Dict[str, str]],
                  **kwargs
                  ):
        return plot_sampling(sampling_flux_df=flux_df_dic,
                             rxn_ids=rxn_ids,
                             **kwargs)


class rFastCormicThresholdPlotter(BasePlotter):
    def __init__(self, dpi=150, prefix="curve_"):
        super().__init__(dpi, prefix)

    def plot_func(self, x, y,
                  exp_th, nonexp_th,
                  right_curve=None, left_curve=None, *args, **kwargs):
        return plot_rFastCormic_thresholds(x,
                                           y,
                                           exp_th,
                                           nonexp_th,
                                           right_c=right_curve,
                                           left_c=left_curve)


class PercentileThresholdPlotter(BasePlotter):
    def __init__(self, dpi=150, prefix="threshold_"):
        super().__init__(dpi, prefix)

    def plot_func(self, data, exp_th, *args, **kwargs):

        return plot_percentile_thresholds(data=data,
                                          exp_th=exp_th, *args, **kwargs)


class ComponentComparisonPlotter(BasePlotter):
    def __init__(self, dpi=150, prefix="component_"):
        super().__init__(dpi, prefix)

    def plot_func(self,
                  result,
                  xticklabels=True,
                  yticklabels=True,
                  scale=1,
                  cbar_label='Jaccard Index',
                  cmap='magma',
                  *args,
                  **kwargs):
        return plot_heatmap(data=result,
                            xticklabels=xticklabels,
                            yticklabels=yticklabels,
                            scale=scale,
                            cbar_label=cbar_label,
                            cmap=cmap,
                            **kwargs)


class DimReductionPlotter(BasePlotter):
    def __init__(self, dpi=150, prefix="dim_reduction_"):
        super().__init__(dpi, prefix)

    def plot_func(self,
                  method,
                  flux_df,
                  groups,
                  *args,
                  **kwargs):
        if method == "PCA":
            plot_PCA(flux_df,
                     groups=groups,
                     *args,
                     **kwargs)
        else:
            plot_embedding(flux_df,
                           groups=groups,
                           reducer=method,
                           *args,
                           **kwargs)