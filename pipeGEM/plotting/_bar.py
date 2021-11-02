import matplotlib.pyplot as plt
import seaborn as sns

from pipeGEM.plotting._utils import save_fig


@save_fig
def plot_model_components(new_comp_df,
                          order,
                          **kwargs):
    """

    Parameters
    ----------
    new_comp_df
    order
    kwargs

    Returns
    -------

    """
    fig_titles = ["reactions", "metabolites", "genes"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 7))
    order_key = {v: i for i, v in enumerate(order)}
    new_comp_df = new_comp_df.sort_values(by=["group"], key=lambda x: x.apply(lambda x1: order_key[x1]))
    sns.boxplot(data=new_comp_df[new_comp_df["component"] == "reactions"], y="number", x="group", hue="group",
                palette="deep", order=order, ax=axes[0], dodge=False)

    sns.boxplot(data=new_comp_df[new_comp_df["component"] == "metabolites"], y="number", x="group", hue="group",
                palette="deep", order=order, ax=axes[1], dodge=False)

    sns.boxplot(data=new_comp_df[new_comp_df["component"] == "genes"], y="number", x="group", hue="group",
                palette="deep", order=order, ax=axes[2], dodge=False)

    for i in range(3):
        axes[i].get_legend().remove()
        axes[i].set_title(fig_titles[i])
        if i != 0:
            axes[i].set_ylabel("")

    return {"g": fig}