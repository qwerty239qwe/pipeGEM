from typing import List, Union, Dict
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ._utils import save_fig, draw_significance
from ._prep import prep_fva_plotting_data, prep_flux_df, filter_fva_df
from pipeGEM.analysis import StatisticAnalyzer


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
        all_zeros_rxn = flux_df.loc[:, (abs(flux_df) < threshold).all(axis=0)].columns.to_list()
        if verbosity > 0:
            print(f"Reactions contain zeros fluxes: {all_zeros_rxn}, were removed from the plot")
        flux_df = flux_df.loc[:, (abs(flux_df) > threshold).any(axis=0)]
    flux_df = pd.concat([flux_df, model_df.loc[:, [group_layer]]], axis=1)
    flux_df = flux_df.melt(id_vars=[group_layer],
                           var_name=["Reactions"],
                           value_name=r'Flux ($\mu$mol/min/gDW)')
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
    if filter_all_zeros:
        min_flux_df, max_flux_df = filter_fva_df(min_df=min_flux_df, max_df=max_flux_df,
                                                 threshold=threshold, verbosity=verbosity)
    ready_df = prep_fva_plotting_data(min_flux_df, max_flux_df, model_df_max[group_layer])
    fig, ax = plt.subplots()
    sns.boxplot(data=ready_df,
                x="Reactions" if vertical else "Flux",
                y="Flux" if vertical else "Reactions",
                hue="model",
                ax=ax,
                whis=10,
                linewidth=0)
    ax.set_title(fig_title)
    plot_kws = {"g": fig}
    if name_format:
        for k, v in kwargs.items():
            if isinstance(v, str):
                plot_kws[k] = v
        plot_kws["name_format"] = name_format

    return plot_kws


@save_fig(dpi=300)
def plot_one_sampling(flux_df,
                      r: str,
                      color_maps,
                      grps,
                      file_dir,
                      group_layer: str = "",
                      plotting_style='displot',
                      plotting_kind='kde',
                      plot_significance = True,
                      fig_title="Flux Sampling: {r}",
                      prefix: str = "FS_",
                      **kwargs):
    facet: sns.FacetGrid = getattr(sns, plotting_style)(data=flux_df,
                                                        y=r,
                                                        x=group_layer if plotting_style == "catplot" else None,
                                                        hue=group_layer if plotting_style == "displot" else None,
                                                        kind=plotting_kind,
                                                        palette=color_maps,
                                                        alpha=1 / len(grps),
                                                        **kwargs)
    ax = facet.ax
    ax.set_title(fig_title.format(r))
    if plotting_kind in ["kde", "hist"]:
        for grp, color in color_maps.items():
            ax.axvline(np.median(flux_df[r]),
                       color=color,
                       linestyle='--',
                       label=f'median ({grp})')
    elif plotting_style == "catplot":
        if plot_significance:
            grp_df = flux_df[[r, group_layer]]
            grp_df["n"] = grp_df.index
            grp_df.pivot(index="n", columns=group_layer)
            stat = StatisticAnalyzer(grp_df)
            do_comp = True
            num_significance = 0
            if grp_df.shape[1] >= 3:
                stat_value, p_value = stat.kruskal_test()
                do_comp = (p_value < stat.alpha_list[0])  # p < 0.05
            if do_comp:
                post_hoc_df = stat.post_hocs()
                for i, j in itertools.combinations(range(grp_df.shape[1]), 2):
                    stars = sum([post_hoc_df.iloc[i, j] < alpha for alpha in stat.alpha_list])
                    if stars >= 1:
                        draw_significance(ax, [i, j],
                                          [max(grp_df[r]) +
                                           (ax.get_ylim()[1] - ax.get_ylim()[0]) *
                                           (num_significance + 1) * 0.08
                                           for _ in range(2)],
                                          stars)
                        num_significance += 1
    plot_kws = {"file_dir": str(file_dir),
                "g": facet.figure,
                "rxn_id": r,
                "name_format": "{file_dir}/{rxn_id}.png"}
    return plot_kws


def plot_sampling(sampling_flux_df: Dict[str, pd.DataFrame],  # n_samples: (n_models, n_rxns)
                  sample_category_df: pd.DataFrame,  # (n_samples, n_cats)
                  rxn_ids: Union[List[str], Dict[str, str]],
                  group_layer: str = "",
                  plotting_style='displot',
                  plotting_kind='kde',
                  palette: str = "muted",
                  plot_significance = True,
                  fig_title="Flux Sampling: {r}",
                  file_dir="./sampling",
                  prefix: str = "FS_",
                  **kwargs):
    flux_df = pd.concat([sampling_flux_df, sample_category_df], axis=1)
    if isinstance(rxn_ids, dict):
        flux_df = flux_df.rename(columns=rxn_ids)
        rxn_ids = list(rxn_ids.values())
    flux_df = flux_df.loc[:, rxn_ids + sample_category_df.columns.to_list()]
    grps = flux_df[group_layer].unique()
    color_maps = {g: sns.color_palette(palette, n_colors=len(grps)) for g in grps}
    for r in rxn_ids:
        plot_one_sampling(flux_df=flux_df,
                          r=r,
                          color_maps=color_maps,
                          grps=grps,
                          group_layer=group_layer,
                          plotting_style=plotting_style,
                          plotting_kind=plotting_kind,
                          plot_significance=plot_significance,
                          file_dir=file_dir,
                          fig_title=fig_title,
                          prefix=prefix,
                          **kwargs
                          )
