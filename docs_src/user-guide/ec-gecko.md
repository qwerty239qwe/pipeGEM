# EC and GECKO Models

pipeGEM includes enzyme-constrained integration helpers under `pipeGEM.integration.ec`.

The public entry points are:

- `apply_gecko_light`
- `apply_gecko_full`
- `auto_parameterize`

Use these workflows when reaction capacity should be constrained by enzyme usage, molecular weight, kcat, or automated parameterization data.

```python
from pipeGEM.integration.ec import apply_gecko_light

ec_model = apply_gecko_light(model)
```

GECKO and enzyme-constrained workflows depend on enzyme metadata quality. Check reaction-gene mappings, enzyme molecular weights, kcat coverage, and units before interpreting flux differences.
