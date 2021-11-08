from typing import List, Union, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ._utils import save_fig
from ._prep import prep_fva_plotting_data, prep_flux_df


@save_fig(dpi=300)
def plot_fba(flux_df: pd.DataFrame,
             rxn_ids: Union[List[str], Dict[str, str]],
             kind: str = "bar",
             palette: Union[str] = "deep",
             group_layer: str = "",
             filter_all_zeros: bool = True,
             fig_title: str = None,
             threshold: float = 1e-6,
             vertical: bool = True,
             name_format: str = "{method}_result.png",
             verbosity: int = 0,
             **kwargs
             ):
    model_df, flux_df = prep_flux_df(flux_df, rxn_ids)
    if filter_all_zeros:
        all_zeros_rxn = (abs(flux_df) < threshold).all(axis=0).columns.to_list()
        if verbosity > 0:
            print(f"Reactions contain zeros fluxes: {all_zeros_rxn}, were removed from the plot")
        flux_df = flux_df.loc[:, (abs(flux_df) > threshold).any(axis=0)]
    flux_df = pd.concat([flux_df, model_df], axis=1)
    flux_df = flux_df.melt(id_vars=model_df.columns,
                           var_name=["Reactions"],
                           value_name=[r'Flux ($\mu$mol/min/gDW)'])
    x_var = "Reactions" if vertical else r'Flux ($\mu$mol/min/gDW)'
    y_var = "Reactions" if not vertical else r'Flux ($\mu$mol/min/gDW)'
    g = sns.catplot(data=flux_df,
                    x=x_var,
                    y=y_var,
                    hue=group_layer,
                    kind=kind,
                    palette=palette,
                    height=6, aspect=2)
    if fig_title is not None:
        g.set_title(fig_title)
    plot_kws = {"g": g}
    if name_format:
        for k, v in kwargs.items():
            if isinstance(v, str):
                plot_kws[k] = v
        plot_kws["name_format"] = name_format
    # Show Fig
    return plot_kws


@save_fig(dpi=300)
def plot_fva(min_flux_df: pd.DataFrame,
             max_flux_df: pd.DataFrame,
             rxn_ids: Union[List[str], Dict[str, str]],
             group_layer: str = "",
             fig_title: str = None,
             filter_all_zeros: bool = True,
             threshold: float = 1e-6,
             vertical: bool = True,
             name_format: str = "{method}_result.png",
             verbosity: int = 0,
             **kwargs
             ):
    model_df_min, min_flux_df = prep_flux_df(min_flux_df, rxn_ids)
    model_df_max, max_flux_df = prep_flux_df(max_flux_df, rxn_ids)
    ready_df = prep_fva_plotting_data(min_flux_df, max_flux_df)
    fig, ax = plt.subplots()
    sns.boxplot(data=ready_df, x="Reactions",
                y="Flux", hue="model",
                ax=ax,
                whis=10, linewidth=0)
    ax.set_title(fig_title)
    plot_kws = {"g": fig}
    if name_format:
        for k, v in kwargs.items():
            if isinstance(v, str):
                plot_kws[k] = v
        plot_kws["name_format"] = name_format

    return plot_kws


def plot_sampling():
    pass

