import pandas as pd
import cobra
import pytest

from pipeGEM.core import Model
from pipeGEM.core import Group


def test_PCA(pFBA_result):
    pca = pFBA_result.dim_reduction()
    assert pca is not None
    assert hasattr(pca, "embedding_df")
    assert isinstance(pca.embedding_df, pd.DataFrame)
    assert pca.embedding_df.shape[1] >= 2  # at least 2 components


def test_TSNE(pFBA_result):
    tsne = pFBA_result.dim_reduction(method="TSNE", n_components=2)
    assert tsne is not None
    assert hasattr(tsne, "embedding_df")
    assert isinstance(tsne.embedding_df, pd.DataFrame)
    assert tsne.embedding_df.shape[1] == 2


def test_UMAP(pFBA_result):
    umap_res = pFBA_result.dim_reduction(n_neighbors=2,
                                         method="UMAP",
                                         n_components=2)
    assert umap_res is not None
    assert hasattr(umap_res, "embedding_df")
    assert isinstance(umap_res.embedding_df, pd.DataFrame)
    assert umap_res.embedding_df.shape[1] == 2


def test_corr(pFBA_result):
    corr_result = pFBA_result.corr(group_by="treatments")
    assert corr_result is not None
    assert hasattr(corr_result, "corr_df")
    corr_df = corr_result.corr_df
    assert isinstance(corr_df, pd.DataFrame)
    # Correlation matrix should be square
    assert corr_df.shape[0] == corr_df.shape[1]
    # Values should be in [-1, 1]
    assert corr_df.min().min() >= -1.0 - 1e-10
    assert corr_df.max().max() <= 1.0 + 1e-10


def test_corr_rxn_corr(pFBA_result):
    corr_result = pFBA_result.corr(group_by="model", rxn_corr=True)
    assert corr_result is not None
    assert hasattr(corr_result, "corr_df")
    assert isinstance(corr_result.corr_df, pd.DataFrame)


def test_flux_plot_heatmap_default(pFBA_result):
    result = pFBA_result.plot_heatmap()
    # plot_heatmap may return None or a plot object; at least it shouldn't error
