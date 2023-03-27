import pandas as pd
import numpy as np


def get_syn_gene_data(model,
                      n_sample,
                      n_genes = None,
                      groups = None,
                      random_state = 42):
    genes = [g.id for g in model.genes]
    rng = np.random.default_rng(random_state)
    if n_genes is not None:
        used_genes = genes[:n_genes]
        if len(used_genes) < n_genes:
            used_genes += [f"not_metabolic_gene_{i + 1}" for i in range(n_genes - used_genes)]
    else:
        n_genes = len(genes)
        used_genes = genes
    return pd.DataFrame(data=np.clip(np.concatenate([rng.negative_binomial(100, rng.uniform(0.01, 1), (1, n_sample)) +
                                                     rng.normal(0, rng.uniform(1, 50), (1, n_sample))
                                             for _ in range(n_genes)], axis=0), a_min=0, a_max=None),
                        columns=[f"sample_{i}" for i in range(n_sample)],
                        index=used_genes)