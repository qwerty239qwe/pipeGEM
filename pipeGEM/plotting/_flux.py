from typing import List, Union, Dict
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ._utils import save_fig, draw_significance
from ._prep import prep_fva_plotting_data, filter_fva_df
# from pipeGEM.analysis import StatisticAnalyzer


def plot_fba(flux_df: pd.DataFrame,
             rxn_ids: Union[List[str], Dict[str, str]],
             kind: str = "bar",
             palette: Union[str] = "deep",
             model_hue: bool = False,
             filter_all_zeros: bool = True,
             fig_title: str = None,
             threshold: float = 1e-6,
             vertical: bool = True,
             height: int = 6,
             aspect: int = 2,
             name_format: str = "{method}_result.png",
             flux_unit: str = "($m$mol/hr/gDW)",
             verbosity: int = 0,
             **kwargs
             ):
    flux_df = flux_df.loc[flux_df["Reaction"].isin(rxn_ids), [c for c in flux_df.columns if c != "reduced_costs"]]
    if filter_all_zeros:
        n_all_zeros_rxn = flux_df.query(f"fluxes < {threshold}").shape[0]
        if verbosity > 0:
            print(f"Found reactions contain zeros fluxes: {n_all_zeros_rxn} rxns were removed from the plot")
        flux_df = flux_df.query(f"fluxes > {threshold}")

    flux_df = flux_df.reset_index().rename(columns={"fluxes": f'Flux {flux_unit}'})
    x_var = "Reaction" if vertical else f'Flux {flux_unit}'
    y_var = "Reaction" if not vertical else f'Flux {flux_unit}'
    g = sns.catplot(data=flux_df,
                    x=x_var,
                    y=y_var,
                    hue="name" if model_hue else None,
                    kind=kind,
                    palette=palette,
                    height=height,
                    aspect=aspect)
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


def plot_fva(fva_df: pd.DataFrame,
             rxn_ids: Union[List[str], Dict[str, str]],
             fig_title: str = None,
             filter_all_zeros: bool = True,
             model_hue: bool = False,
             threshold: float = 1e-6,
             vertical: bool = True,
             name_format: str = "{method}_result.png",
             verbosity: int = 0,
             **kwargs
             ):
    fva_df = fva_df.loc[rxn_ids, :]
    if filter_all_zeros:
        fva_df = filter_fva_df(fva_df=fva_df, threshold=threshold, verbosity=verbosity)
    ready_df = prep_fva_plotting_data(fva_df)
    fig, ax = plt.subplots()
    sns.boxplot(data=ready_df,
                x="Reactions" if vertical else "Flux",
                y="Flux" if vertical else "Reactions",
                hue="name" if model_hue else None,
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
    facet = getattr(sns, plotting_style)(data=flux_df,
                                         x=r if plotting_style != "catplot" else group_layer,
                                         y= None if plotting_style != "catplot" else r,
                                         hue=group_layer if plotting_style == "displot" else None,
                                         kind=plotting_kind,
                                         palette=color_maps,
                                         **kwargs)
    ax = facet.ax
    ax.set_title(fig_title.format(r=r))
    if plotting_kind in ["kde", "hist"]:
        for grp, color in color_maps.items():
            ax.axvline(np.median(flux_df[r]),
                       color=color,
                       linestyle='--',
                       label=f'median ({grp})')
    elif plotting_style == "catplot":
        # if plot_significance:
        #     grp_df = flux_df[[r, group_layer, "n"]]
        #     grp_df = grp_df.pivot(index="n", columns=group_layer).T.reset_index().set_index("group").drop(columns=["level_0"]).T
        #     stat = StatisticAnalyzer(grp_df)
        #     do_comp = True
        #     num_significance = 0
        #     if grp_df.shape[1] >= 3:
        #         stat_value, p_value = stat.kruskal_test()
        #         print(f"Kruskal Wallis test result: stat: {stat_value}, p-value {p_value}")
        #         do_comp = (p_value < stat.alpha_list[0])  # p < 0.05
        #     if do_comp:
        #         post_hoc_df = stat.post_hocs()
        #         for i, j in itertools.combinations(range(grp_df.shape[1]), 2):
        #             stars = sum([post_hoc_df.iloc[i, j] < alpha for alpha in stat.alpha_list])
        #             if stars >= 1:
        #                 draw_significance(ax, [i, j],
        #                                   [grp_df.values.max() +
        #                                    (ax.get_ylim()[1] - ax.get_ylim()[0]) *
        #                                    (num_significance + 1) * 0.08
        #                                    for _ in range(2)],
        #                                   stars)
        #                 num_significance += 1
        pass
    plot_kws = {
                "g": facet.fig,
               } # TODO: fix saving function (add name_format)
    return plot_kws


def plot_sampling(sampling_flux_df: Dict[str, pd.DataFrame],  # n_samples: (n_models, n_rxns)
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
    dfs = []
    for n, df in sampling_flux_df.items():
        df["n"] = n
        dfs.append(df)

    flux_df = pd.concat(dfs, axis=0, ignore_index=True)
    if isinstance(rxn_ids, dict):
        flux_df = flux_df.rename(columns=rxn_ids)
        rxn_ids = list(rxn_ids.values())
    flux_df = flux_df.reindex(columns=rxn_ids + [group_layer, "n"])
    grps = flux_df[group_layer].unique()
    pl = sns.color_palette(palette, n_colors=len(grps))
    color_maps = {g: pl[i] for i, g in enumerate(grps)}
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
