from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pipeGEM.plotting._utils import _set_default_ax


def plot_PCA(data,
             groups = None,
             title=None,
             plot_2D=True,
             plot_score=True,
             plot_scree=True,
             plot_loading=False,
             sheet_file_name=None,
             score_prefix="PC_",
             scree_prefix="scree_",
             loading_prefix="loading_",
             **kwargs):
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


def plot_embedding(embedding_df,
                   groups: dict = None,
                   title=None,
                   palette="muted",
                   plot_2D=True,
                   reducer="UMAP",
                   figsize=(7, 7),
                   sheet_file_name=None,
                   **kwargs
                   ):
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


def plot_3D_PCA_score(pca_df,
                      groups,
                      palette="muted",
                      fig_title=None,
                      **kwargs):
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


def plot_PCA_screeplot(exp_var_df,
                       palette="muted",
                       fig_title=None,
                       **kwargs):

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