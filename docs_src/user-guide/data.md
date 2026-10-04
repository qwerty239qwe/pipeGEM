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
3. Set `absent_expression` to represent measured genes at or below `expression_threshold`.
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

Reaction mapping follows the model's parsed GPR tree, including parentheses.
The defaults take the minimum for `and` and the maximum for `or`: with
`g1=2`, `g2=5`, and `g3=10`, `g1 and (g2 or g3)` scores **2**.
For merged reactions, `plus_operation="nansum"` combines the component scores.

Genes absent from the dataset use `missing_value` (default `NaN`). The default
`nanmin` and `nanmax` reductions ignore missing values when another value is
available; use `and_operation="min"` to propagate a missing complex subunit.
Empty rules and rules with no finite scores return the missing value. Scores
at or below `threshold` become `absent_value`. `data_transform` is applied
when accessing `GeneData.rxn_scores`, after mapping.

`gene_data.rxn_mapper.partial_map(model, updated_data, changed_gene_ids)`
recomputes reactions involving the changed genes. Supply a complete updated
`GeneData` dataset, including unchanged genes. Mapping options from the initial
alignment are retained unless overridden for that update.

## Fetching and synthesis

The `pipeGEM.data.fetching` helpers load remote models and public data where supported. The `pipeGEM.data.synthesis` helpers generate synthetic data for examples and tests.

Use synthetic data only for demonstrations or smoke tests. For biological analysis, document the source database, normalization method, and gene identifier namespace alongside the generated model.

See [Data API](../api/data.md).
