from typing import List, Union, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ._utils import save_fig


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
    model_df = flux_df.loc[:, [i for i in flux_df.columns if i not in rxn_ids]]
    flux_df = flux_df.copy().loc[:, rxn_ids]
    if filter_all_zeros:
        all_zeros_rxn = (abs(flux_df) < threshold).all(axis=0).columns.to_list()
        if verbosity > 0:
            print(f"Reactions contain zeros fluxes: {all_zeros_rxn}, were removed from the plot")
        flux_df = flux_df.loc[:, (abs(flux_df) > threshold).any(axis=0)]
    n_comps, n_rxns = flux_df.shape
    n_grps = len(model_df[group_layer].unique())
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
def plot_fva(flux_df: pd.DataFrame,
             rxn_ids: Union[List[str], Dict[str, str]],
             kind: str = "bar",
             group_layer: str = "",
             filter_all_zeros: bool = True,
             fig_title: str = None,
             threshold: float = 1e-6,
             vertical: bool = True,
             name_format: str = "{method}_result.png",
             verbosity: int = 0,
             ):
    pass

def plot_sampling():
    pass

