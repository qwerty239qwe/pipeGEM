from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Optional, Union, List

from pipeGEM.plotting._utils import _set_default_ax


def plot_PCA(data: Dict[str, pd.DataFrame],
             groups: Optional[Dict[str, Union[str, List[str]]]] = None,
             title: Optional[str] = None,
             plot_2D: bool = True,
             plot_score: bool = True,
             plot_scree: bool = True,
             plot_loading: bool = False,
             sheet_file_name: Optional[str] = None,
             score_prefix: str = "PC_",
             scree_prefix: str = "scree_",
             loading_prefix: str = "loading_",
             **kwargs):
    """
    Generate various plots for a principal component analysis (PCA) of the given data.

    Parameters:
    -----------
    data: dict
        A dictionary containing the following keys:
            - 'PC': pandas DataFrame with the principal components as columns
            - 'exp_var': pandas DataFrame with the explained variance of each principal component
            - 'components': pandas DataFrame with the loadings of each principal component
    groups: dict, optional
        A dictionary mapping group names to lists of column names to group together in the plots.
        By default, each column is assigned to its own group.
    title: str, optional
        A title to add to the plots. If not provided, a default title is used.
    plot_2D: bool, optional
        If True, generate a 2D plot of the principal component scores. Defaults to True.
    plot_score: bool, optional
        If True, generate a plot of the principal component scores. Defaults to True.
    plot_scree: bool, optional
        If True, generate a plot of the explained variance for each principal component. Defaults to True.
    plot_loading: bool, optional
        If True, generate a plot of the component loadings for each principal component. Defaults to False.
    sheet_file_name: str, optional
        If provided, the generated plots are saved as CSV files with the same name as this string,
        but with "_PCA", "_EXP_VAR", or "_COMP" appended to the filename for the PCA, explained variance,
        and components DataFrames, respectively.
    score_prefix: str, optional
        If provided, a prefix to add to the filename of the score plot when saving to disk. Defaults to "PC_".
    scree_prefix: str, optional
        If provided, a prefix to add to the filename of the scree plot when saving to disk. Defaults to "scree_".
    loading_prefix: str, optional
        If provided, a prefix to add to the filename of the loading plot when saving to disk. Defaults to "loading_".
    **kwargs: additional keyword arguments
        Additional keyword arguments to pass to the individual plotting functions.

    Returns:
    --------
    A dictionary with no keys or values, used solely as a placeholder to indicate that the function has completed.
    """
    pca_df, exp_var_df, component_df = data["PC"], data["exp_var"], data["components"]

    if groups is None:
        groups = {m: [m] for m in pca_df.columns}

    if sheet_file_name is not None:
        sh_path = Path(sheet_file_name)
        pca_path = (sh_path.parent / Path(str(sh_path.stem) + "_PCA")).with_suffix(sh_path.suffix)
        expvar_path = (sh_path.parent / Path(str(sh_path.stem) + "_EXP_VAR")).with_suffix(sh_path.suffix)
        comp_path = (sh_path.parent / Path(str(sh_path.stem) + "_COMP")).with_suffix(sh_path.suffix)
        pca_df.to_csv(pca_path)
        exp_var_df.to_csv(expvar_path)
        component_df.to_csv(comp_path)

    if plot_scree:
        screeplot_kw = {k: v for k, v in kwargs}
        if "file_name" in screeplot_kw:
            screeplot_kw["file_name"] = scree_prefix + screeplot_kw["file_name"]
        plot_PCA_screeplot(exp_var_df,
                           fig_title=f"{title} scree plot" if title is not None else "Scree plot",
                           **screeplot_kw)

    if plot_loading:
        loading_kw = {k: v for k, v in kwargs}
        if "file_name" in loading_kw:
            loading_kw["file_name"] = loading_prefix + loading_kw["file_name"]
        plot_PCA_loading(component_df,
                         fig_title=f"{title} loading plot" if title is not None else "Loading plot",
                         **loading_kw)

    pc_kw = {k: v for k, v in kwargs}
    if "file_name" in pc_kw:
        pc_kw["file_name"] = score_prefix + pc_kw["file_name"]
    if plot_2D and plot_score:

        plot_2D_PCA_score(pca_df, groups,
                          fig_title=f"{title} score plot" if title is not None else "PC Score plot",
                          exp_var_df=exp_var_df,
                          **pc_kw)
    elif plot_score:
        plot_3D_PCA_score(pca_df, groups,
                          fig_title=f"{title} score plot" if title is not None else "PC Score plot",
                          exp_var_df=exp_var_df,
                          **pc_kw)
    return {}


def plot_embedding(embedding_df: pd.DataFrame,
                   groups: dict = None,
                   title: str = None,
                   palette: str = "muted",
                   plot_2D: bool = True,
                   reducer: str = "UMAP",
                   figsize: tuple = (7, 7),
                   sheet_file_name: str = None,
                   **kwargs: dict
                   ) -> dict:
    """
    Plot embeddings in 2D or 3D.

    Parameters:
    -----------
    embedding_df : pd.DataFrame
        A pandas DataFrame with the embeddings. The rows should be named "embedding 1", "embedding 2", and "embedding 3"
        if plotting in 3D. The columns should be named after the samples or models.
    groups : dict, optional
        A dictionary with group names as keys and lists of model names as values.
    title : str, optional
        A title for the plot.
    palette : str, optional
        A seaborn color palette for the groups.
    plot_2D : bool, optional
        If True, plot the embeddings in 2D. If False, plot in 3D.
    reducer : str, optional
        The dimensionality reduction method used to generate the embeddings.
    figsize : tuple, optional
        A tuple with the figure size (width, height) in inches.
    sheet_file_name : str, optional
        A file name to save the embedding_df as a csv file.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the plotting functions.

    Returns:
    --------
    plotting_kws : dict
        A dictionary with keyword arguments passed to the plotting function.
    """
    plotting_kws = {k: v for k, v in kwargs.items()
                    if k in ["file_name", "dpi", "prefix"]} if kwargs is not None else {}
    for k in ["file_name", "dpi", "prefix"]:
        if k in kwargs:
            del kwargs[k]
    colors = sns.color_palette(palette)
    fig, ax = plt.subplots(figsize=figsize)
    if sheet_file_name is not None:
        embedding_df.to_csv(sheet_file_name)

    if groups is None:
        groups = {m: [m] for m in embedding_df.columns}

    for i, (group_name, model_names) in enumerate(groups.items()):
        em1, em2 = np.array([embedding_df.loc['embedding 1', name] for name in model_names]), \
                   np.array([embedding_df.loc['embedding 2', name] for name in model_names])

        if plot_2D:
            ax.scatter(em1, em2, s=50, label=group_name, c=[colors[i]])
        else:
            em3 = np.array([embedding_df.loc['embedding 3', name] for name in model_names])
            ax.scatter(em1, em2, em3, s=50, label=group_name, c=[colors[i]])

    x_label = f'{reducer} 1'
    y_label = f'{reducer} 2'

    ax = _set_default_ax(ax, x_label=x_label, y_label=y_label, title=title)

    plotting_kws.update({"g": fig})
    return plotting_kws


def plot_2D_PCA_score(pca_df,
                      groups = None,
                      features: dict = None,
                      colors=None,
                      palette="muted",
                      continuous=False,
                      fig_title=None,
                      **kwargs):
    """
    Plot a scatter plot of 2D PCA scores.

    Parameters:
    -----------
    pca_df : pd.DataFrame
        A DataFrame containing 2D PCA scores.
    groups : dict
        A dictionary of group names as keys and lists of model names as values.
    features : dict
        A dictionary of feature names as keys and corresponding column names as values.
    colors : list of str
        A list of color names or hex codes to use for each group.
    palette : str
     A string indicating the color palette to use if colors is not provided.
    continuous : bool
        A boolean indicating whether to use a continuous colormap.
    fig_title : str
        A string indicating the title of the figure.
    kwargs:
        Additional keyword arguments to be passed to the function.

    Returns:
    --------
    plotting_kws : dict
        A dictionary of keyword arguments to be passed to the plotting function.
    """

    colors = sns.color_palette(palette,
                               as_cmap=continuous) if colors is None else colors
    fig, ax = plt.subplots(figsize=(7, 7))
    if features is None:
        for i, (group_name, model_names) in enumerate(groups.items()):
            pc1, pc2 = np.array([pca_df.loc['PC1', name] for name in model_names]), \
                       np.array([pca_df.loc['PC2', name] for name in model_names])

            ax.scatter(pc1, pc2, label=group_name, c=[colors[i]])
    else:
        x = pca_df.loc['PC1', :].values
        y = pca_df.loc['PC2', :].values
        z = pca_df.columns.map(features)
        points = ax.scatter(x, y, c=z, s=50, cmap=colors)
        fig.colorbar(points)

    ax.axvline(0, color=(0.1, 0.1, 0.1, 0.7))
    ax.axhline(0, color=(0.1, 0.1, 0.1, 0.7))
    pc1_exp, pc2_exp = None, None
    if 'exp_var_df' in kwargs:
        pc1_exp, pc2_exp = kwargs['exp_var_df'].iloc[0, 0], kwargs['exp_var_df'].iloc[1, 0]
    x_label = f'Principle Component 1' + f'(explain {pc1_exp * 100:.2f}% variance)' if pc1_exp else ''
    y_label = f'Principle Component 2' + f'(explain {pc2_exp * 100:.2f}% variance)' if pc2_exp else ''
    title = f'PCA_score{("_" + fig_title) if fig_title is not None else ""}'
    ax = _set_default_ax(ax, x_label=x_label, y_label=y_label, title=title)
    plotting_kws = {k: v for k, v in kwargs.items() if k in ["file_name", "dpi", "prefix"]}
    plotting_kws.update({"g": fig})
    return plotting_kws


def plot_3D_PCA_score(pca_df: pd.DataFrame,
                      groups: dict,
                      palette: str = "muted",
                      fig_title: str = None,
                      **kwargs) -> dict:
    """
    Plot 3D scatter plot of PCA scores.

    Parameters:
    -----------
    pca_df : pd.DataFrame
        Dataframe containing the PCA scores for each sample.
    groups : dict
        Dictionary where the keys represent the group name and the values are lists of model names that belong to that
        group.
    palette : str, default="muted"
        The color palette for the plot.
    fig_title : str, default=None
        The title for the plot.
    **kwargs :
        Additional keyword arguments to pass to the plot.

    Returns:
    --------
    dict
        A dictionary containing the plot parameters.

    """
    colors = sns.color_palette(palette)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    for i, (group_name, model_names) in enumerate(groups.items()):
        pc1, pc2, pc3 = np.array([pca_df.loc['PC1', name] for name in model_names]), \
                        np.array([pca_df.loc['PC2', name] for name in model_names]), \
                        np.array([pca_df.loc['PC3', name] for name in model_names])

        ax.scatter(pc1, pc2, pc3, label=group_name, c=[colors[i]])

    pc1_exp, pc2_exp, pc3_exp = None, None, None
    if 'exp_var_df' in kwargs:
        pc1_exp, pc2_exp, pc3_exp = kwargs['exp_var_df'].iloc[0, 0], kwargs['exp_var_df'].iloc[1, 0], kwargs['exp_var_df'].iloc[2, 0]
    x_label = f'PC1' + f'(explain {pc1_exp * 100:.2f}% variance)' if pc1_exp else ''
    y_label = f'PC2' + f'(explain {pc2_exp * 100:.2f}% variance)' if pc2_exp else ''
    z_label = f'PC3' + f'(explain {pc3_exp * 100:.2f}% variance)' if pc3_exp else ''
    title = f'PCA_score{("_" + fig_title) if fig_title is not None else ""}'

    ax = _set_default_ax(ax, x_label=x_label, y_label=y_label, z_label=z_label, title=title)
    plotting_kws = {k: v for k, v in kwargs.items() if k in ["file_name", "dpi", "prefix"]}
    plotting_kws.update({"g": fig})
    return plotting_kws


def plot_PCA_screeplot(exp_var_df: pd.DataFrame,
                       palette: str = "muted",
                       fig_title: str = None,
                       **kwargs) -> dict:
    """
    Creates a scree plot showing the variance explained by each principle component.

    Parameters:
    -----------
    exp_var_df : pandas DataFrame
        A DataFrame containing the proportion of variance explained by each principle component.
        The index of the DataFrame should be the name of the principle component (e.g., "PC1").
        The DataFrame should have a single column containing the proportion of variance explained.
    palette : str or sequence of colors, optional (default="muted")
        Color palette to use for the plot.
    fig_title : str or None, optional (default=None)
        Title of the plot.
    **kwargs : dict
        Additional keyword arguments to be passed to the underlying Matplotlib functions.

    Returns:
    --------
    dict
        A dictionary containing the keyword arguments for plotting the figure.
        This dictionary can be passed to the `savefig` function to save the figure to disk.
    """
    colors = sns.color_palette(palette)
    exp_var = exp_var_df.values
    cumsum = np.cumsum(exp_var)

    fig, ax = plt.subplots(figsize=(11 * len(exp_var) / 12 if len(exp_var) > 12 else 11, 8))
    X = np.arange(len(exp_var))
    ax.bar(X, exp_var.flatten(), color=colors[0], width=.4)
    ax.plot(X, cumsum, marker='o', color=colors[1], label="Cumulative (%)")

    ax = _set_default_ax(ax, title=f'{fig_title}' if fig_title is not None else fig_title,
                         x_label='Principle Component',
                         y_label='Variance Explaned (%)')

    ax.set_xticks(X)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xticklabels([f'PC{x + 1}' for x in range(len(X))])
    ax.set_yticklabels(np.linspace(0, 100, 11))
    ax.grid(True, axis='y')
    plt.xlim([-1, len(X)])
    plt.ylim([0, 1.1])
    plotting_kws = {k: v for k, v in kwargs.items() if k in ["file_name", "dpi", "prefix"]}
    plotting_kws.update({"g": fig})
    return plotting_kws


def plot_PCA_loading(component_df: pd.DataFrame,
                     n_feature=30,
                     fig_title=None,
                     fig_size=(11, 11),
                     **kwargs):
    """
    Plot the PCA loading plot for top `n_feature` features.

    Parameters:
    -----------
    component_df: pd.DataFrame
        The dataframe containing PCA component scores.
    n_feature: int, optional (default=30)
        Number of features to plot in the loading plot.
    fig_title: str, optional (default=None)
        Title of the plot.
    fig_size: tuple, optional (default=(11, 11))
        Figure size.
    **kwargs:
        Additional keyword arguments to be passed to the plotting function.

    Returns:
    --------
    dict:
        Dictionary containing the plot object and the keyword arguments for saving the plot.
        """
    plt.style.use("seaborn")
    fig, ax = plt.subplots(figsize=fig_size)
    component_df["key"] = np.sqrt(component_df["PC1"] ** 2 + component_df["PC2"] ** 2)
    component_df = component_df.sort_values("key", ascending=False).drop("key", 1).iloc[:n_feature, :]

    ax = _set_default_ax(ax,
                         title=f'{fig_title}',
                         x_label='PC1',
                         y_label='PC2', with_legend=False)
    for idx, row in component_df.iterrows():
        ax.plot([0, row["PC1"]], [0, row["PC2"]])
        ax.text(x=row["PC1"], y=row["PC2"], s=idx)
    plotting_kws = {k: v for k, v in kwargs.items() if k in ["file_name", "dpi", "prefix"]}
    plotting_kws.update({"g": fig})
    return plotting_kws


def plot_Eflux_scatter(r_exp, r_bound):
    """
    Create a scatter plot showing the relationship between reaction expression levels and lower/upper bounds.

    Parameters
    ----------
    r_exp : dict
        A dictionary containing reaction expression levels. Keys are reaction IDs and values are expression levels.
    r_bound : dict
        A dictionary containing lower and upper bounds for each reaction. Keys are reaction IDs and values are lists
        containing the lower and upper bounds.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))
    axes[0].scatter([r_exp[r_id] for r_id in list(r_exp.keys())],
                    [r_bound[r_id][0] for r_id in list(r_exp.keys())])
    axes[1].scatter([r_exp[r_id] for r_id in list(r_exp.keys())],
                    [r_bound[r_id][1] for r_id in list(r_exp.keys())])
    _set_default_ax(axes[0], title=f'Expression to lower bound',
                    x_label='Expression value',
                    y_label='Lower bound', with_legend=False)

    _set_default_ax(axes[1], title=f'Expression to upper bound',
                    x_label='Expression value',
                    y_label='Upper bound', with_legend=False)
    plt.show()