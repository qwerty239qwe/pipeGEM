# Model and Group Workflows

Use `pipeGEM.Model` for one COBRA model and `pipeGEM.Group` when comparing or analyzing multiple models together.

## Model

`Model` delegates unknown attributes to the wrapped `cobra.Model`, so existing COBRApy workflows continue to work while pipeGEM adds metadata, omics data, medium data, tasks, and analysis helpers.

```python
import pipeGEM as pg
from pipeGEM.utils import load_model

model = pg.Model(model=load_model("model.xml"), name_tag="sample")
```

Common model operations include:

- Inspect reaction, gene, metabolite, and subsystem IDs.
- Add annotations, gene data, medium data, and metabolic tasks.
- Test consistency and tasks.
- Run flux analysis and knockout analysis.
- Save modified models.

Recommended workflow:

1. Load or construct a `cobra.Model`.
2. Wrap it with a stable `name_tag`; this name appears in grouped results and plots.
3. Add condition-specific data such as expression, medium, enzyme, or task data.
4. Run analysis with an explicit solver, for example `solver="glpk"` for environments without commercial solvers.
5. Inspect the returned result object before saving plots or extracted models.

See [Model API](../api/model.md).

## Group

```python
import pipeGEM as pg

group = pg.Group(
    {"condition_a": {"a1": model_a1}, "condition_b": {"b1": model_b1}},
    name_tag="experiment",
)
```

Use groups to run the same analysis over many models, compare model components, aggregate results, and visualize differences across conditions.

Keep model names unique within the group. Treatment annotations should use the same model names so result tables can be grouped consistently.

See [Group API](../api/group.md).
