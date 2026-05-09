# Data Workflows

pipeGEM data objects prepare measured values for model-aware workflows.

## Gene data

`pipeGEM.data.GeneData` stores expression data, applies transforms, aligns genes with a model, and calculates reaction scores from gene-protein-reaction rules.

```python
from pipeGEM.data import GeneData

gene_data = GeneData(data=expression_series)
model.add_gene_data("sample_1", gene_data)
```

Gene data can then be thresholded or passed into integration algorithms such as GIMME, iMAT, FASTCORE-family methods, RIPTiDe, SPOT, and E-Flux.

Typical preparation steps:

1. Put genes on the index and samples or conditions in columns.
2. Use a transform such as log scaling when the downstream algorithm expects transformed expression.
3. Set `absent_expression` to a value that represents genes missing from the measurement.
4. Check model gene identifiers before integration; mismatched IDs are the most common reason for sparse reaction scores.

```python
import numpy as np
from pipeGEM.data import GeneData

gene_data = GeneData(
    data=expression_series,
    data_transform=lambda values: np.log2(values + 1),
    absent_expression=0,
)

model.add_gene_data("treated_rep1", gene_data)
```

## Fetching and synthesis

The `pipeGEM.data.fetching` helpers load remote models and public data where supported. The `pipeGEM.data.synthesis` helpers generate synthetic data for examples and tests.

Use synthetic data only for demonstrations or smoke tests. For biological analysis, document the source database, normalization method, and gene identifier namespace alongside the generated model.

See [Data API](../api/data.md).
