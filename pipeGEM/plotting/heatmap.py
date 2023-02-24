from typing import List, Dict, Union, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from ._utils import save_fig, _get_subsystem_ticks


def plot_heatmap(data: Union[pd.DataFrame, np.ndarray],
                 scale: int = 1,
                 cbar_label: str = '',
                 cbar_kw: Dict[str, Any] = None,
                 annotate: bool = True,
                 fig_title: Optional[str] = None,
                 **kwargs) -> Dict[str, plt.Figure]:
    """
    Plot a heatmap of the input data and (optional) save it.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        The rectangular dataset to plot.
    scale : int, optional
        An integer representing the scaling factor for the heatmap, by default 1.
    cbar_label : str, optional
        A string representing the label for the colorbar, by default ''.
    cbar_kw : dict, optional
        A dictionary containing additional keyword arguments to be passed to the colorbar, by default None.
    annotate : bool, optional
        A boolean value indicating whether to annotate the heatmap with the data values, by default True.
    fig_title : str, optional
        A string representing the title of the figure, by default None.
    **kwargs : optional
        Additional keyword arguments to be passed to seaborn.heatmap().

    Returns
    -------
    dict
        A dictionary containing the plot figure.
    """
    if cbar_kw is None:
        cbar_kw = {}
    cbar_kw.update({"label": cbar_label})
    grid_kws = {"width_ratios": (.9, .05), "wspace": .3}
    fig, (ax, cbar_ax) = plt.subplots(1, 2, figsize=(data.shape[0] * scale, data.shape[1] * scale), gridspec_kw=grid_kws)
    ax = sns.heatmap(data,
                     ax=ax,
                     cbar_ax=cbar_ax,
                     cbar_kws=cbar_kw,
                     annot=annotate,
                     **kwargs
                     )
    ax.set_title(fig_title)
    plotting_kws = {"g": fig}
    return plotting_kws


def plot_heatmap2(data: Union[pd.DataFrame, np.ndarray],
                  scale: int = 1,
                  row_color_dict = None,
                  row_color_order = None,
                  col_color_dict = None,
                  col_color_order = None,
                  cbar_label: str = '',
                  cbar_kw: Dict[str, Any] = None,
                  row_palette: str = "muted",
                  col_palette: str = "muted",
                  annotate: bool = True,
                  fig_title: str = None,
                  ):
    if cbar_kw is None:
        cbar_kw = {}
    cbar_kw.update({"label": cbar_label})
    grid_kws = {"width_ratios": (.9, .05), "wspace": .3}
    fig, (ax, cbar_ax) = plt.subplots(1, 2, figsize=(data.shape[0] * scale, data.shape[1] * scale),
                                      gridspec_kw=grid_kws)

    if row_color_dict is not None:
        row_colors = sns.color_palette(row_palette)


    ax.set_title(fig_title)
    plotting_kws = {"g": fig}
    return plotting_kws

def _modify_clustermap_for_subsys(g, ticks_pos, subsystems):
    g.ax_heatmap.yaxis.set_ticks(ticks_pos)
    g.ax_heatmap.yaxis.set_ticklabels(subsystems)
    g.ax_heatmap.yaxis.set_ticks_position("left")
    offset = mpl.transforms.ScaledTranslation(-0.3, 0, g.fig.dpi_scale_trans)
    for label in g.ax_heatmap.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)


def plot_clustermap(
        data: pd.DataFrame,
        model_groups: Optional[Dict[str, str]] = None,
        group_list: Optional[List[str]] = None,
        row_category: Optional[str] = None,
        cbar_label: Optional[str] = None,
        top_ele_ratio: float = 0.1,
        row_cluster: bool = False,
        row_dendrogram: bool = True,
        palette: str = "muted",
        c_palette: str = "Spectral",
        fig_title: Optional[str] = None,
        fig_size: Tuple[float, float] = (10, 40),
        **kwargs) -> Dict[str, plt.Figure]:
    """
    Plot a clustered heatmap of the input data and (optional) save it.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Rectangular dataset to plot.
    model_groups : dict or None, optional
        Dictionary mapping sample ids to group labels used to color the columns.
    group_list : list or None, optional
        List of group labels used to color the columns. If None, use all unique values of `model_groups`.
    row_category : pd.Series or None, optional
        Pandas series that contains a categorical variable used to color the rows.
    cbar_label : str or None, optional
        Colorbar label. If None, no colorbar label is shown.
    top_ele_ratio : float, optional
        Ratio of the top elements to the heatmap, given as a float in [0, 1].
    row_cluster : bool, optional
        Whether to cluster the rows. Default is False.
    row_dendrogram : bool, optional
        Whether to show the dendrogram of the rows. Default is True.
    palette : str or list-like, optional
        Palette name (as a string) or list of colors to use for the model groups. Default is "muted".
    c_palette : str, optional
        Color palette name (as a string) used to color the rows based on `row_category`.
        Default is "Spectral".
    fig_title : str or None, optional
        Figure title. If None, no title is shown.
    fig_size : tuple, optional
        Figure size in inches, as a tuple of width and height. Default is (10, 40).
    **kwargs
        Additional keyword arguments passed to `sns.clustermap`.

    Returns
    -------
    plotting_kws : dict
        A dictionary containing the resulting matplotlib figure as a value under the key "g".
    """
    col_colors, row_colors = None, None
    if model_groups and group_list:
        colors = sns.color_palette(palette)
        color_dict = dict(zip(group_list, colors[:len(group_list)]))
        col_colors = [color_dict[model_groups[c]] for c in data.columns]

    nonzeros = (data.var(axis=1) != 0)
    data = data.loc[nonzeros, :]
    if row_category is not None:
        data, subsystems, ticks_pos = _get_subsystem_ticks(data, row_category)
        colors = sns.color_palette(c_palette, as_cmap=True)(np.linspace(0, 1, len(subsystems)))
        color_dict = dict(zip(subsystems, colors))
        row_colors = {r: color_dict[row_category[r]] for r in data.index}
        row_colors = [color for r_id, color in row_colors.items()]

    g = sns.clustermap(data=data,
                       figsize=fig_size,
                       row_cluster=row_cluster,
                       fmt=".3f",
                       col_colors=col_colors,
                       row_colors=row_colors,
                       cbar_pos=(0, 1-top_ele_ratio+0.01, 0.05, top_ele_ratio-0.01),
                       dendrogram_ratio=top_ele_ratio,
                       colors_ratio=(0.02, 0.4 / fig_size[1]),
                       cbar_kws={"label": cbar_label},
                       **kwargs)
    plt.title(fig_title)
    if row_colors is not None and not row_cluster:
        _modify_clustermap_for_subsys(g, ticks_pos, subsystems)

    if not row_dendrogram:
        g.ax_row_dendrogram.remove()

    plotting_kws = {"g": g}
    return plotting_kws