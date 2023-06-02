from enum import Enum
from typing import Optional, Callable

import umap
from sklearn.decomposition import PCA, IncrementalPCA
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


def prepare_PCA_dfs(feature_df: pd.DataFrame,
                    transform_func: Optional[Callable] = None,
                    n_components: Optional[int] = None,
                    standardize: bool = True,
                    incremental: bool = False):
    """
    Prepare principal component analysis (PCA) dataframes from a feature dataframe.

    Parameters
    ----------
    feature_df: pd.DataFrame
        The feature dataframe to analyze. Rows represent features, and columns represent samples.
    transform_func: callable, optional
        A function to apply to the feature dataframe before analysis. Default is None.
    n_components: int, optional
        The number of components in the result dataframes. If None, the minimum of the feature_df's
        shape[0] and shape[1] is used. Default is None.
    standardize: bool
        Whether to standardize the feature dataframe before analysis by centering and scaling to unit variance.
        Default is True.
    incremental: bool
        Whether to use an incremental PCA algorithm instead of a regular PCA algorithm.
        Default is False.

    Returns
    -------
    PC_df: pd.DataFrame
        A dataframe containing the principal components (columns) of each sample (rows).
    exp_var_df: pd.DataFrame
        A dataframe containing the explained variance ratio of each principal component (rows).
    component_df: pd.DataFrame
        A dataframe containing the principal axes in feature space, representing the directions of maximum variance
        in the data.

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
    if incremental:
        pca = IncrementalPCA(n_components=n_components)
    else:
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


def prepare_embedding_dfs(feature_df: pd.DataFrame,
                          transform_func: Optional[Callable] = None,
                          n_components: int = 3,
                          reducer: str = "TSNE",
                          standardize: bool = True,
                          **kwargs):
    """
    Get a dataframe containing an embedding result from a feature dataframe.

    Parameters
    ----------
    feature_df: pd.DataFrame
        A dataframe where the rows are features and the columns are samples.
    transform_func: callable, optional
        A function that will be performed on the dataframe before analysis.
    n_components: int, optional
        Number of components in the result dfs.
    reducer: str or Reducer enum member, optional
        A string or enum specifying the dimensionality reduction algorithm to use.
        Supported options are "TSNE", "Isomap", "MDS", "SpectralEmbedding", "LocallyLinearEmbedding", and "UMAP".
        Default is "TSNE".
    standardize: bool, optional
        If True, standardize the dataframe before analysis by removing the mean and scaling to unit variance.
        Default is True.
    **kwargs: dict
        Additional keyword arguments to be passed to the dimensionality reduction algorithm.

    Returns
    -------
    df: pd.DataFrame
        The embedding result containing the component values of each data (rows).
        The index of the returned dataframe is the embedding component number (e.g., "embedding 1", "embedding 2").
        The columns are the sample names from the input feature_df.

    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html
    https://umap-learn.readthedocs.io/en/latest/
    """
    if transform_func:
        x = transform_func(feature_df.values)
    else:
        x = feature_df.values
    if isinstance(reducer, str):
        reducer = Reducer(reducer)
    sample_names = feature_df.columns.to_list()
    x = StandardScaler().fit_transform(x.T) if standardize else x.values.T
    if reducer == Reducer("TSNE") and kwargs.get("perplexity") is None:
        kwargs["perplexity"] = min(x.shape[0]-1, 30)
    X_embedded = REDUCER_DICT[reducer](n_components=n_components, **kwargs).fit_transform(x)
    df = pd.DataFrame(X_embedded,
                      columns=["embedding {}".format(i) for i in range(1, n_components + 1)],
                      index=sample_names)
    return df
