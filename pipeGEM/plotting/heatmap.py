from typing import List, Dict, Union, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from ._utils import save_fig, _get_subsystem_ticks


@save_fig
def plot_heatmap(data: Union[pd.DataFrame, np.ndarray],
                 scale: int = 1,
                 cbar_label: str = '',
                 cbar_kw: Dict[str, Any] = None,
                 annotate: bool = True,
                 fig_title: str = None,
                 **kwargs):
    """
    Plot a heatmap of the input data and (optional) save it.

    Parameters
    ----------
    data: rectangular dataset

    scale
    cbar_label
    cbar_kw
    annotate
    fig_title
    kwargs

    Returns
    -------

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


def _modify_clustermap_for_subsys(g, ticks_pos, subsystems):
    g.ax_heatmap.yaxis.set_ticks(ticks_pos)
    g.ax_heatmap.yaxis.set_ticklabels(subsystems)
    g.ax_heatmap.yaxis.set_ticks_position("left")
    offset = mpl.transforms.ScaledTranslation(-0.3, 0, g.fig.dpi_scale_trans)
    for label in g.ax_heatmap.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)


@save_fig(prefix="clustermap_", dpi=150)
def plot_clustermap(data,
                    model_groups=None,
                    group_list=None,
                    row_category=None,
                    cbar_label=None,
                    top_ele_ratio=0.1,
                    row_cluster=False,
                    row_dendrogram=True,
                    palette="muted",
                    c_palette="Spectral",
                    fig_title=None,
                    fig_size=(10, 40),
                    **kwargs):
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