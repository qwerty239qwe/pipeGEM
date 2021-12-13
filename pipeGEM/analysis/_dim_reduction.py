from enum import Enum

import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, MDS, SpectralEmbedding, LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
import pandas as pd


class Reducer(Enum):
    TSNE = "TSNE"
    Isomap = "Isomap"
    MDS = "MDS"
    SpectralEmbedding = "SpectralEmbedding"
    LocallyLinearEmbedding = "LocallyLinearEmbedding"
    UMAP = "UMAP"


REDUCER_DICT = {Reducer.TSNE: TSNE,
                Reducer.MDS: MDS,
                Reducer.Isomap: Isomap,
                Reducer.LocallyLinearEmbedding: LocallyLinearEmbedding,
                Reducer.SpectralEmbedding: SpectralEmbedding,
                Reducer.UMAP: umap.UMAP}


def prepare_PCA_dfs(feature_df,
                    transform_func=None,
                    n_components=None,
                    standardize=True):
    """
    Get three dataframes containing PCA results from a feature dataframe

    Parameters
    ----------
    feature_df: a pd.DataFrame
        The feature dataframe, the rows are the features of each data
    transform_func: optional, callable
        A function that will be performed on the dataframe before analysis
    n_components: optional, int
        Number of components in the result dfs, if None than the minimal number of df.shape[0] and df.shape[1] is used.
    standardize: bool
        If true, standardize the dataframe before the analysis by removing the mean and scaling to unit variance.

    Returns
    -------
    PC_df: pd.DataFrame
        The PCA result containing the PCs (columns) values of each data (rows)
    exp_var_df: pd.DataFrame
        The dataframe containing explained_variance_ratio_ of each PC (rows)
    component_df: pd.DataFrame
        The dataframe containing principal axes in feature space,
        representing the directions of maximum variance in the data.

    References
    -----------
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    """
    if transform_func is not None:
        x = transform_func(feature_df)
    else:
        x = feature_df
    x = StandardScaler().fit_transform(x.values.T) if standardize else x.values.T

    if not n_components:
        n_components = min(x.shape[0], x.shape[1])
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(x)
    final_df = pd.DataFrame(data=principal_components,
                            columns=[f'PC{num + 1}' for num in range(principal_components.shape[1])],
                            index=feature_df.columns).T
    exp_var_df = pd.DataFrame(data=pca.explained_variance_ratio_,
                              index=[f'PC{num + 1}' for num in range(n_components)])
    component_df = pd.DataFrame(data=pca.components_.T,
                                columns=[f'PC{num + 1}' for num in range(n_components)],
                                index=feature_df.index)
    return final_df, exp_var_df, component_df


def prepare_embedding_dfs(feature_df,
                          transform_func=None,
                          n_components=2,
                          reducer="TSNE",
                          standardize=True, **kwargs):
    """

    Parameters
    ----------
    feature_df
    transform_func
    n_components
    reducer
    standardize
    kwargs

    Returns
    -------

    """
    if transform_func:
        x = transform_func(feature_df.values)
    else:
        x = feature_df.values
    if isinstance(reducer, str):
        reducer = Reducer(reducer)
    sample_names = feature_df.columns.to_list()
    x = StandardScaler().fit_transform(x.T) if standardize else x.values.T
    X_embedded = REDUCER_DICT[reducer](n_components=n_components, **kwargs).fit_transform(x)
    df = pd.DataFrame(X_embedded,
                      columns=["embedding {}".format(i) for i in range(1, n_components + 1)],
                      index=sample_names)
    return df.T
