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
             group_by=None,
             kind: str = "bar",
             palette: Union[str] = "deep",
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
    if "Reaction" not in flux_df.columns:
        flux_df = flux_df.reset_index().rename(columns={"index": "Reaction"})
        print("Use index as the reaction IDs")
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
                    hue=group_by,
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


def plot_sampling_displot(flux_df,
                          rxn_id,
                          kind,
                          group_by,
                          vertical,
                          stat_analysis=None):
    xy_val = {"x": rxn_id if vertical else None,
              "y": rxn_id if not vertical else None}
    facet = sns.displot(data=flux_df,
                        x=xy_val["x"],
                        y=xy_val["y"],
                        kind=kind,
                        hue=group_by)
    if stat_analysis is not None:
        pass

    return facet


def plot_sampling_catplot(flux_df,
                          rxn_id,
                          kind,
                          group_by,
                          group_order,
                          vertical,
                          stat_analysis=None,
                          stat_analysis_rxn_id_col="label",
                          star_notation_cutoffs=None):
    xy_val = {"x": rxn_id if not vertical else group_by,
              "y": rxn_id if vertical else group_by}
    if group_order is None:
        group_order = sorted(flux_df[group_by].unique())

    facet = sns.catplot(data=flux_df,
                        x=xy_val["x"],
                        y=xy_val["y"],
                        kind=kind,
                        hue=group_by,
                        dodge=False,
                        order=group_order)
    if stat_analysis is not None:
        if star_notation_cutoffs is None:
            star_notation_cutoffs = [0.05,] + [10**(-i) for i in range(2, 5)]

        num_significance = 0

        for ia, ib in itertools.combinations(range(len(group_order)), 2):
            sig_df = stat_analysis.result_df
            sig_df = sig_df[sig_df[stat_analysis_rxn_id_col] == rxn_id]
            sig_df = sig_df[((sig_df["A"] == group_order[ia]) & (sig_df["B"] == group_order[ib]) |
                             (sig_df["B"] == group_order[ia]) & (sig_df["A"] == group_order[ib]))]

            n_stars = sum([sig_df["adjusted_p_value"].values[0] < alpha for alpha in star_notation_cutoffs])
            if n_stars < 1:
                continue

            if vertical:
                x_pos = [ia, ib]
                y_pos = [flux_df[rxn_id].values.max() + (facet.ax.get_ylim()[1] - facet.ax.get_ylim()[0]) *
                         (num_significance + 1) * 0.08
                         for _ in range(2)]
            else:
                x_pos = [flux_df[rxn_id].values.max() + (facet.ax.get_xlim()[1] - facet.ax.get_xlim()[0]) *
                         (num_significance + 1) * 0.1
                         for _ in range(2)]
                print(x_pos)
                y_pos = [ia, ib]

            draw_significance(facet.ax, x_pos, y_pos, n_stars)
            num_significance += 1
        if vertical:
            facet.set(ylim=(facet.ax.get_ylim()[0], facet.ax.get_ylim()[1] +
                             (facet.ax.get_ylim()[1] - facet.ax.get_ylim()[0]) *
                             (num_significance + 1) * 0.08))
        else:
            facet.set(xlim=(facet.ax.get_xlim()[0], facet.ax.get_xlim()[1] +
                             (facet.ax.get_xlim()[1] - facet.ax.get_xlim()[0]) *
                             (num_significance + 1) * 0.1))


    return facet


def plot_sampling_df(flux_df,
                     rxn_id,
                     kind,
                     group_by,
                     group_order=None,
                     vertical=True,
                     plotting_type = "displot",
                     stat_analysis = None,
                     ):
    assert plotting_type in ["displot", "catplot"]

    if plotting_type == "displot":
        facet = plot_sampling_displot(flux_df=flux_df,
                                      rxn_id=rxn_id,
                                      kind=kind,
                                      group_by=group_by,
                                      vertical=vertical,
                                      stat_analysis=stat_analysis)
    else:
        facet = plot_sampling_catplot(flux_df=flux_df,
                                      rxn_id=rxn_id,
                                      kind=kind,
                                      group_by=group_by,
                                      group_order=group_order,
                                      vertical=vertical,
                                      stat_analysis=stat_analysis)
    fig_kws = {"g": facet.figure}

    return fig_kws


#  the codes below are deprecated
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
