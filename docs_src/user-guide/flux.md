# Flux Analysis

Use `Model.do_flux_analysis` or `Group.do_flux_analysis` to run flux workflows and return result objects with plotting and downstream-analysis methods.

```python
result = model.do_flux_analysis("pFBA", solver="glpk")
result.plot(rxn_ids=["EX_glc__D_e", "BIOMASS"])
```

Common result objects include FBA, pFBA-style flux analysis, FVA, sampling, knockout, correlation, and dimensionality-reduction analyses.

Run flux analysis after applying condition-specific media, integration results, or task-derived model changes when those constraints are part of the experiment.

For reproducible runs, record the solver, objective reaction, medium constraints, and any reaction bounds changed before analysis. Small bound changes can dominate flux results even when the model topology is unchanged.

See [Results API](../api/results.md).
