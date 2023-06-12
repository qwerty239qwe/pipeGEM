import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Optional, Tuple, Union


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
        axes[i].get_legend().remove()
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
        ax.axhline(y=global_on_th.loc[g], xmin=block_w * gpi, xmax=block_w * (gpi + 1),
                   label="Global-on threshold", lw=2, ls=":", color="k")
        ax.axhline(y=global_off_th.loc[g], xmin=block_w * gpi, xmax=block_w * (gpi + 1),
                   label="Global-off threshold", lw=2, ls=":", color="b")
        local_width_init = block_w * gpi + gap
        for gni, gene in enumerate(genes):
            ax.axhline(y=local_th.loc[gene, g],
                       xmax=local_width_init + actual_width * (gni + 1),
                       xmin=local_width_init + actual_width * gni,
                       label="Local threshold", lw=2, ls="-", color="r")
    ax.set_xlim(*x_lims)

    return {"g": fig}