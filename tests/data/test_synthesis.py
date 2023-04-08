from scipy.sparse import issparse
from anndata import AnnData
import numpy as np
from pandas.api.types import is_numeric_dtype
from pipeGEM.data.synthesis import get_syn_gene_data


def test_get_syn_gene_data_dataframe(ecoli_core):
    # test returned_dtype='DataFrame'
    df = get_syn_gene_data(ecoli_core, n_sample=100, n_genes=len(ecoli_core.genes), random_state=42)
    assert df.shape == (len(ecoli_core.genes), 100)
    assert (df.dtypes == np.float64).all()

    assert len(set(df.index) - set([g.id for g in ecoli_core.genes])) == 0
    assert all(df.columns == [f"sample_{i}" for i in range(100)])


def test_get_syn_gene_data_anndata(ecoli_core):
    # test returned_dtype='AnnData'
    ad = get_syn_gene_data(ecoli_core, n_sample=100, n_genes=len(ecoli_core.genes), random_state=42, returned_dtype="AnnData")
    assert isinstance(ad, AnnData)
    assert ad.X.shape == (100, len(ecoli_core.genes))
    assert issparse(ad.X) == False
    assert all(ad.obs.index == [f"sample_{i}" for i in range(100)])
    assert len(set(ad.var.index) - set([g.id for g in ecoli_core.genes])) == 0


def test_get_syn_gene_data_all_genes(ecoli_core):
    # test with n_genes=None (use all genes)
    df = get_syn_gene_data(ecoli_core, n_sample=100, n_genes=None, random_state=42)
    assert df.shape == (len(ecoli_core.genes), 100)
    assert (df.dtypes == np.float64).all()
    assert len(set(df.index) - set([g.id for g in ecoli_core.genes])) == 0
    assert all(df.columns == [f"sample_{i}" for i in range(100)])


def test_get_syn_gene_data_more_genes_than_model(ecoli_core):
    # test with n_genes > number of genes in model
    df = get_syn_gene_data(ecoli_core, n_sample=100, n_genes=len(ecoli_core.genes) + 10, random_state=42)
    assert df.shape == (len(ecoli_core.genes) + 10, 100)
    assert (df.dtypes == np.float64).all()
    assert len(set(df.index) - (set([f"not_metabolic_gene_{i + 1}" for i in range(10)]) |
               set([g.id for g in ecoli_core.genes]))) == 0
    assert all(df.columns == [f"sample_{i}" for i in range(100)])