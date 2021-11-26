import pandas as pd
import numpy as np


def get_syn_gene_data(model,
                      n_sample,
                      n_genes = None,
                      groups = None,
                      random_state = 34):
    genes = [g.id for g in model.genes]
    rng = np.random.default_rng(random_state)
    if n_genes is not None:
        used_genes = genes[:n_genes]
        if len(used_genes) < n_genes:
            used_genes += [f"not_metabolic_gene_{i + 1}" for i in range(n_genes - used_genes)]
    else:
        n_genes = len(genes)
        used_genes = genes
    return pd.DataFrame(data=rng.negative_binomial(200000, 0.98, (n_genes, n_sample)),
                        columns=[f"sample_{i}" for i in range(n_sample)],
                        index=used_genes)