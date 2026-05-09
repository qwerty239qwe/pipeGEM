from typing import List, Union, Dict
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ._utils import save_fig, draw_significance
from ._prep import prep_fva_plotting_data, filter_fva_df
# from pipeGEM.analysis import StatisticAnalyzer
from pipeGEM._logging import get_logger

logger = get_logger(__name__)


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
    """Plot FBA flux results as a categorical plot.

    Parameters
    ----------
    flux_df : pd.DataFrame
        DataFrame with a ``"Reaction"`` column and ``"fluxes"`` column (plus
        optional categorical columns like ``"model"``).
    rxn_ids : list of str or dict
        Reaction IDs to plot.  If a dict, keys are original IDs and values
        are display names.
    group_by : str, optional
        Column name used to colour/group the bars (e.g., ``"model"``).
    kind : str, optional
        Seaborn catplot kind — ``"bar"``, ``"box"``, ``"violin"``, etc.
    palette : str, optional
        Seaborn palette name.
    filter_all_zeros : bool, optional
        Remove reactions whose absolute flux is below *threshold*.
    fig_title : str, optional
        Figure title.
    threshold : float, optional
        Minimum absolute flux for a reaction to be plotted.
    vertical : bool, optional
        If *True*, reactions on x-axis, fluxes on y-axis.
    height, aspect : int, optional
        Seaborn FacetGrid sizing parameters.
    name_format : str, optional
        Template for auto-generated file names.
    flux_unit : str, optional
        Unit label appended to the flux axis.

    Returns
    -------
    dict
        ``{"g": FacetGrid, "name_format": str, ...}``
    """
    if "Reaction" not in flux_df.columns:
        flux_df = flux_df.reset_index().rename(columns={"index": "Reaction"})
        logger.debug("Use index as the reaction IDs")
    flux_df = flux_df.loc[flux_df["Reaction"].isin(rxn_ids), [c for c in flux_df.columns if c != "reduced_costs"]]
    if filter_all_zeros:
        n_all_zeros_rxn = flux_df.query(f"abs(fluxes) < {threshold}").shape[0]
        if verbosity > 0:
            logger.info("Found reactions contain zeros fluxes: %d rxns were removed from the plot", n_all_zeros_rxn)
        flux_df = flux_df.query(f"abs(fluxes) > {threshold}")

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
        g.figure.suptitle(fig_title)
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
             color_by: bool = "model",
             threshold: float = 1e-6,
             vertical: bool = True,
             name_format: str = "{method}_result.png",
             verbosity: int = 0,
             **kwargs
             ):
    """Plot FVA flux ranges as a boxplot.

    Parameters
    ----------
    fva_df : pd.DataFrame
        DataFrame with ``"Reaction"``, ``"minimum"`` and ``"maximum"`` columns.
    rxn_ids : list of str or dict
        Reactions to include.
    fig_title : str, optional
        Figure title.
    filter_all_zeros : bool, optional
        Remove reactions with ``|max − min| < threshold``.
    color_by : str, optional
        Column used for hue grouping, by default ``"model"``.
    threshold : float, optional
        Minimum range for a reaction to be plotted.
    vertical : bool, optional
        If *True*, reactions on x-axis.
    name_format : str, optional
        File-name template.

    Returns
    -------
    dict
        ``{"g": Figure, "name_format": str, ...}``
    """
    fva_df = fva_df.loc[fva_df["Reaction"].isin(rxn_ids), :]
    if filter_all_zeros:
        fva_df = filter_fva_df(fva_df=fva_df, threshold=threshold, verbosity=verbosity)
    ready_df = prep_fva_plotting_data(fva_df)
    fig, ax = plt.subplots()
    sns.boxplot(data=ready_df,
                x="Reaction" if vertical else "Flux",
                y="Flux" if vertical else "Reaction",
                hue=color_by,
                ax=ax,
                whis=10,
                width=0.6,
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
                          stat_analysis=None,
                          y_lim=None,
                          x_lim=None,
                          **kwargs):
    xy_val = {"x": rxn_id if vertical else None,
              "y": rxn_id if not vertical else None}
    facet = sns.displot(data=flux_df,
                        x=xy_val["x"],
                        y=xy_val["y"],
                        kind=kind,
                        hue=group_by,
                        **kwargs)
    if y_lim is not None:
        facet.set(ylim=y_lim)
    if x_lim is not None:
        facet.set(xlim=x_lim)

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
                          star_notation_cutoffs=None,
                          y_lim=None,
                          x_lim=None,
                          **kwargs):
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
                        order=group_order,
                        **kwargs)
    if stat_analysis is not None:
        if star_notation_cutoffs is None:
            star_notation_cutoffs = [0.05,] + [10**(-i) for i in range(2, 5)]

        num_significance = 0

        for ia, ib in itertools.combinations(range(len(group_order)), 2):
            sig_df = stat_analysis.result_df
            sig_df = sig_df[sig_df[stat_analysis_rxn_id_col] == rxn_id]
            sig_df = sig_df[((sig_df["A"] == group_order[ia]) & (sig_df["B"] == group_order[ib]) |
                             (sig_df["B"] == group_order[ia]) & (sig_df["A"] == group_order[ib]))]
            if sig_df.shape[0] == 0:
                raise ValueError(f"Cannot find the comparison: {group_order[ia]} vs {group_order[ib]}",
                                 "Changing the group_by according to the original analysis might solve the problem.")

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
                logger.debug("x_pos: %s", x_pos)
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
    if y_lim is not None:
        logger.debug("y_lim: %s", y_lim)
        facet.set(ylim=y_lim)
    if x_lim is not None:
        facet.set(xlim=x_lim)

    return facet


def plot_sampling_df(flux_df,
                     rxn_id,
                     kind,
                     group_by,
                     group_order=None,
                     vertical=True,
                     plotting_type = "displot",
                     stat_analysis = None,
                     y_lim=None,
                     x_lim=None,
                     **kwargs
                     ):
    """Plot flux sampling results as a distribution or categorical plot.

    Parameters
    ----------
    flux_df : pd.DataFrame
        Sampling flux DataFrame.
    rxn_id : str
        Reaction column to plot.
    kind : str
        Seaborn plot kind (``"kde"``, ``"hist"``, ``"box"``, ``"violin"``, …).
    group_by : str
        Column used for grouping / hue.
    group_order : list of str, optional
        Order of groups on the categorical axis.
    vertical : bool, optional
        Orientation.
    plotting_type : {``"displot"``, ``"catplot"``}
        Which seaborn high-level function to use.
    stat_analysis : object, optional
        A pairwise test result used to draw significance stars (catplot only).
    y_lim, x_lim : tuple, optional
        Axis limits.

    Returns
    -------
    dict
        ``{"g": Figure}``
    """
    assert plotting_type in ["displot", "catplot"]

    if plotting_type == "displot":
        facet = plot_sampling_displot(flux_df=flux_df,
                                      rxn_id=rxn_id,
                                      kind=kind,
                                      group_by=group_by,
                                      vertical=vertical,
                                      stat_analysis=stat_analysis,
                                      y_lim=y_lim,
                                      x_lim=x_lim,
                                      **kwargs)
    else:
        facet = plot_sampling_catplot(flux_df=flux_df,
                                      rxn_id=rxn_id,
                                      kind=kind,
                                      group_by=group_by,
                                      group_order=group_order,
                                      vertical=vertical,
                                      stat_analysis=stat_analysis,
                                      y_lim=y_lim,
                                      x_lim=x_lim,
                                      **kwargs)
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
