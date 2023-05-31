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
    fig, axes = plt.subplots(1, 3, figsize=(12, 7))
    order_key = {v: i for i, v in enumerate(order)}
    comp_df = comp_df.sort_values(by=[group], key=lambda x: x.apply(lambda x1: order_key[x1]))
    sns.boxplot(data=comp_df[comp_df["component"] == "n_rxns"], y="number", x=group, hue=group,
                palette="deep", order=order, ax=axes[0], dodge=False, **kwargs)
    sns.boxplot(data=comp_df[comp_df["component"] == "n_mets"], y="number", x=group, hue=group,
                palette="deep", order=order, ax=axes[1], dodge=False, **kwargs)
    sns.boxplot(data=comp_df[comp_df["component"] == "n_genes"], y="number", x=group, hue=group,
                palette="deep", order=order, ax=axes[2], dodge=False, **kwargs)
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
                                 figsize: Tuple[float, float] = (8, 6)
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
                ax=ax)

    x_lims = ax.get_xlim()
    for g in groups:
        ax.axhline(y=global_on_th.loc[g], xmax=x_lims[1], label="Global-on threshold", lw=2, ls="--", color="k")
        ax.axhline(y=global_off_th.loc[g], xmax=x_lims[1], label="Global-off threshold", lw=2, ls="--", color="b")

        for gene in genes:
            ax.axhline(y=local_th.loc[gene, g], xmax=x_lims[1], label="Local threshold", lw=2, ls="--", color="r")
    ax.set_xlim(*x_lims)

    return {"g": fig}