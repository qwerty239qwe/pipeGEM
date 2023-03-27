import pandas as pd
import numpy as np
from anndata import AnnData
from typing import Optional, Union
import cobra
from pipeGEM import Model


def get_syn_gene_data(model: Union[cobra.Model, Model],
                      n_sample: int,
                      n_genes: Optional[int] = None,
                      groups: Optional[str] = None,
                      random_state: int = 42,
                      returned_dtype: str = "DataFrame"
                      ) -> Union[pd.DataFrame, AnnData]:
    """
    Generate synthetic gene expression data with a given number of samples and genes.

    Parameters
    ----------
    model : cobra.Model or pg.Model
        A model containing information about the genes to simulate expression data for.
    n_sample : int
        The number of samples to generate.
    n_genes : int, optional
        The number of genes to simulate expression data for. If None, use all genes in the model (default=None).
    groups : str, optional
        The name of the attribute containing group information for the genes (default=None).
    random_state : int, optional
        The random seed to use for generating the data (default=42).
    returned_dtype : str, optional
        The type of object to return. Must be either 'DataFrame' or 'AnnData' (default='DataFrame').

    Returns
    -------
    Union[pd.DataFrame, AnnData]
        The simulated gene expression data. If returned_dtype is 'DataFrame',
        returns a pandas DataFrame with gene IDs as the index and sample IDs as the columns.
        If returned_dtype is 'AnnData', returns an AnnData object with the simulated expression data as the X attribute,
        and empty obs and var attributes.
    """
    assert returned_dtype in ["DataFrame", "AnnData"]

    genes = [g.id for g in model.genes]
    rng = np.random.default_rng(random_state)

    if n_genes is not None:
        used_genes = genes[:n_genes]
        if len(used_genes) < n_genes:
            used_genes += [f"not_metabolic_gene_{i + 1}" for i in range(n_genes - len(used_genes))]
    else:
        n_genes = len(genes)
        used_genes = genes

    data = np.clip(np.concatenate([
        rng.negative_binomial(100, rng.uniform(0.01, 1), (1, n_sample)) +
        rng.normal(0, rng.uniform(1, 50), (1, n_sample))
        for _ in range(n_genes)], axis=0), a_min=0, a_max=None)

    if returned_dtype == "DataFrame":
        return pd.DataFrame(data=data, columns=[f"sample_{i}" for i in range(n_sample)], index=used_genes)
    elif returned_dtype == "AnnData":
        return AnnData(X=data.T, obs=pd.DataFrame(index=[f"sample_{i}" for i in range(n_sample)]),
                       var=pd.DataFrame(index=used_genes))