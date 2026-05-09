# Medium Catalog

pipeGEM includes medium definitions in `pipeGEM.data.medium` and exposes medium utilities through `pipeGEM.data.MediumData` and the medium registry.

Built-in medium files include common recipes such as LB, M9, M63, MOPS, DMEM, DMEM high FFA, serum, BHI, SOC, TB, R2A, Hams, and CGXII.

## Apply medium data

```python
from pipeGEM.data import MediumData

medium = MediumData.from_file("medium.tsv")
model.add_medium_data("condition", medium)
model.apply_medium("condition")
```

Use medium constraints before flux analysis, model testing, or context-specific reconstruction when the biological condition requires a defined nutrient environment.
