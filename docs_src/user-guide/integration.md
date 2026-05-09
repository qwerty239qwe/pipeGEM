# Gene Data Integration

pipeGEM supports context-specific model generation from expression data through `Model.integrate_gene_data` and lower-level integration functions.

## Typical flow

1. Load a model.
2. Add `GeneData`.
3. Choose and calculate thresholds.
4. Run an integration algorithm.
5. Inspect the returned analysis object and extracted model.

```python
result = model.integrate_gene_data(
    data_name="sample_1",
    integrator="GIMME",
    high_exp=1.0,
)

context_model = result.result_model
```

Choose an algorithm based on the information available:

- Use threshold-driven methods such as GIMME or iMAT when expression values can be split into active and inactive sets.
- Use core-reaction methods such as FASTCORE, SWIFTCORE, CORDA, or MBA when a trusted core set is available.
- Use continuous methods such as E-Flux, SPOT, or RIPTiDe when expression should constrain reaction capacity rather than only include or exclude reactions.

Always inspect the extracted model size, objective feasibility, and task pass rate before using the model in downstream comparisons.

## Supported families

The integration package includes algorithms such as MBA, mCADRE, FASTCORE, rFASTCORMICS, CORDA, iMAT, INIT, RIPTiDe, GIMME, E-Flux, and SPOT.

See [Integration API](../api/integration.md).
