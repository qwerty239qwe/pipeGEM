import pandas as pd
import cobra
import pytest

from pipeGEM.core import Model
from pipeGEM.core import Group


def test_PCA(pFBA_result):
    pca = pFBA_result.dim_reduction()
    pca.plot(dpi=150, color_by="treatments")


def test_TSNE(pFBA_result):
    tsne = pFBA_result.dim_reduction(method="TSNE", n_components=2)
    tsne.plot(dpi=150)


def test_UMAP(pFBA_result):
    tsne = pFBA_result.dim_reduction(n_neighbors=2,
                                     method="UMAP",
                                     n_components=2)
    tsne.plot(dpi=150, color_by="treatments")


def test_corr(pFBA_result):
    corr_result = pFBA_result.corr(group_by="treatments")
    corr_result.plot()


def test_corr_rxn_corr(pFBA_result):
    corr_result = pFBA_result.corr(group_by="model",rxn_corr=True)
    corr_result.plot()


def test_flux_plot_heatmap_default(pFBA_result):
    pFBA_result.plot_heatmap()