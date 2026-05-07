import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Optional, Tuple, Union


def plot_data_cat(long_data: pd.DataFrame,
                  id_col,
                  val_col,
                  ids,
                  group_col=None,
                  vertical=True,
                  kind: str = "bar",
                  palette: Union[str] = "deep",
                  log_scale=False,
                  ):
    """Plot categorical data using seaborn catplot.

    Parameters
    ----------
    long_data : pd.DataFrame
        Long-form DataFrame.
    id_col : str
        Column identifying categories (e.g., reaction IDs).
    val_col : str
        Column with numeric values to plot.
    ids : list of str
        Subset of *id_col* values to include.
    group_col : str, optional
        Column for hue grouping.
    vertical : bool, optional
        If *True*, categories on x-axis and values on y-axis.
    kind : str, optional
        Seaborn catplot kind, by default ``"bar"``.
    palette : str, optional
        Seaborn palette name.
    log_scale : bool, optional
        Apply log scale to the value axis.

    Returns
    -------
    dict
        ``{"g": FacetGrid}``
    """
    if vertical:
        x, y = id_col, val_col
    else:
        x, y = val_col, id_col
    facet = sns.catplot(data=long_data[long_data[id_col].isin(ids)],
                        x=x,
                        y=y,
                        hue=group_col,
                        kind=kind,
                        palette=palette,)
    if log_scale:
        if vertical:
            facet.set(yscale="log")
        else:
            facet.set(xscale="log")
    return {"g": facet}


def plot_model_components(comp_df: pd.DataFrame,
                          order: List[str],
                          group: str = "group",
                          **kwargs):
    """
    Plots boxplots of model components for different groups.

    Parameters
    ----------
    comp_df : pd.DataFrame
        A pandas DataFrame containing information about the model components.
    order : list
        A list containing the order in which the groups should be plotted.
    group : str, optional
        The column name of the grouping variable in the DataFrame, by default "group".
    **kwargs
        Additional keyword arguments to pass to the seaborn boxplot function.

    Returns
    -------
    dict
        A dictionary containing the plot figure.
    """
    fig_titles = ["reactions", "metabolites", "genes"]
    order_key = {v: i for i, v in enumerate(order)}

    if len(order_key) <= 10:
        fig, axes = plt.subplots(1, 3, figsize=(12, 7))
        palette = "deep" if kwargs.get("palette") is None else kwargs["palette"]
    else:
        fig, axes = plt.subplots(3, 1, figsize=(16, 16))
        palette = "Spectral" if kwargs.get("palette") is None else kwargs["palette"]

    comp_df = comp_df.sort_values(by=[group], key=lambda x: x.apply(lambda x1: order_key[x1]))
    sns.boxplot(data=comp_df[comp_df["component"] == "n_rxns"], y="number", x=group, hue=group,
                palette=palette, order=order, ax=axes[0], dodge=False, **kwargs)
    sns.boxplot(data=comp_df[comp_df["component"] == "n_mets"], y="number", x=group, hue=group,
                palette=palette, order=order, ax=axes[1], dodge=False, **kwargs)
    sns.boxplot(data=comp_df[comp_df["component"] == "n_genes"], y="number", x=group, hue=group,
                palette=palette, order=order, ax=axes[2], dodge=False, **kwargs)
    for i in range(3):
        legend = axes[i].get_legend()
        if legend is not None:
            legend.remove()
        axes[i].set_title(fig_titles[i])
        if i != 0:
            axes[i].set_ylabel("")
    return {"g": fig}


def plot_local_threshold_boxplot(data,
                                 genes,
                                 groups,
                                 group_dic,
                                 local_th,
                                 global_on_th,
                                 global_off_th,
                                 width = 0.8,
                                 figsize: Tuple[float, float] = (8, 6),
                                 **kwargs
                                 ):
    """Plot expression boxplots overlaid with local and global thresholds.

    Parameters
    ----------
    data : pd.DataFrame
        Expression matrix (genes × samples).
    genes : list of str
        Gene IDs to include.
    groups : list of str or ``"all"``
        Group names to include (or ``"all"`` for all groups).
    group_dic : dict
        ``{group_name: [sample_names]}``.
    local_th : pd.DataFrame
        Local thresholds (genes × groups).
    global_on_th : pd.Series
        Global "on" threshold per group.
    global_off_th : pd.Series
        Global "off" threshold per group.
    width : float, optional
        Box width, by default 0.8.
    figsize : tuple, optional
        Figure size, by default ``(8, 6)``.

    Returns
    -------
    dict
        ``{"g": Figure}``
    """
    fig, ax = plt.subplots(figsize=figsize)
    if groups == "all":
        groups = [g for g in group_dic]

    selected_samples = [i for g in groups for i in group_dic[g]]
    group_map = {i: g for g in groups for i in group_dic[g]}

    data = data.loc[genes, selected_samples].reset_index().melt(id_vars=["index"],
                                                                var_name="model",
                                                                value_name="expression").rename(columns={"index": "gene"})  # S * G
    data["group"] = data["model"].map(group_map)
    sns.boxplot(data=data,
                y="expression",
                x="group",
                hue="gene",
                hue_order=genes,
                order=groups,
                ax=ax,
                **kwargs)

    x1_, x2_ = len(data["group"].unique()), len(data["gene"].unique())
    block_w = 1 / x1_
    gap = ((1 - width) / x1_) / 2
    actual_width = (1 / x1_ - 2 * gap) / x2_
    x_lims = ax.get_xlim()
    for gpi, g in enumerate(groups):
        # Only add labels on the first group to avoid duplicate legend entries
        on_label = "Global-on threshold" if gpi == 0 else None
        off_label = "Global-off threshold" if gpi == 0 else None
        ax.axhline(y=global_on_th.loc[g], xmin=block_w * gpi, xmax=block_w * (gpi + 1),
                   label=on_label, lw=2, ls=":", color="k")
        ax.axhline(y=global_off_th.loc[g], xmin=block_w * gpi, xmax=block_w * (gpi + 1),
                   label=off_label, lw=2, ls=":", color="b")
        local_width_init = block_w * gpi + gap
        for gni, gene in enumerate(genes):
            local_label = "Local threshold" if gpi == 0 and gni == 0 else None
            ax.axhline(y=local_th.loc[gene, g],
                       xmax=local_width_init + actual_width * (gni + 1),
                       xmin=local_width_init + actual_width * gni,
                       label=local_label, lw=2, ls="-", color="r")
    ax.set_xlim(*x_lims)
    ax.legend()

    return {"g": fig}