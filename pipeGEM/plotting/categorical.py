import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_model_components(comp_df: pd.DataFrame,
                          order,
                          **kwargs):
    """

    Parameters
    ----------
    comp_df
    order
    kwargs

    Returns
    -------

    """
    fig_titles = ["reactions", "metabolites", "genes"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 7))
    order_key = {v: i for i, v in enumerate(order)}
    comp_df = comp_df.sort_values(by=["group"], key=lambda x: x.apply(lambda x1: order_key[x1]))
    sns.boxplot(data=comp_df[comp_df["component"] == "n_rxns"], y="number", x="group", hue="group",
                palette="deep", order=order, ax=axes[0], dodge=False)
    sns.boxplot(data=comp_df[comp_df["component"] == "n_mets"], y="number", x="group", hue="group",
                palette="deep", order=order, ax=axes[1], dodge=False)
    sns.boxplot(data=comp_df[comp_df["component"] == "n_genes"], y="number", x="group", hue="group",
                palette="deep", order=order, ax=axes[2], dodge=False)
    for i in range(3):
        axes[i].get_legend().remove()
        axes[i].set_title(fig_titles[i])
        if i != 0:
            axes[i].set_ylabel("")
    return {"g": fig}