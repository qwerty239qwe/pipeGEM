# Quickstart

pipeGEM is a package for visualizing, analyzing, and integrating data with genome-scale metabolic models. Flux analysis builds on COBRApy, while pipeGEM adds model/group containers, omics-data handling, thresholding, metabolic tasks, medium utilities, and model extraction workflows.

## Command line

Generate template configuration files for a pipeline:

```bash
pipeGEM template -p integration -o ./
```

This writes template TOML files under a `configs` directory. Edit those files to point at your model, gene data, thresholds, mapping tables, and integration settings.

Run model processing:

```bash
pipeGEM process -t configs/model.toml
```

Run expression-data integration:

```bash
pipeGEM integrate \
  -g configs/gene_data.toml \
  -t configs/model.toml \
  -r configs/threshold.toml \
  -m configs/mapping.toml \
  -i configs/integration.toml
```

Run flux analysis:

```bash
pipeGEM flux -f configs/flux_analysis/pFBA.toml -t configs/model.toml
```

Run model comparison:

```bash
pipeGEM compare -c configs/comparison.toml
```

!!! note "Legacy CLI"
    The previous `python -m pipeGEM.cli -n <pipeline>` and `pipeGEM -n <pipeline>` style is still accepted for compatibility, but new scripts should use subcommands.

## Runnable Python example

This example loads the E. coli core model, wraps it in `pipeGEM.Model`, and runs an optimization through the underlying COBRA model.

```python
import pipeGEM as pg
from pipeGEM import load_remote_model

cobra_model = load_remote_model("e_coli_core")
model = pg.Model(name_tag="e_coli_core", model=cobra_model)

print(model)
print(len(model.reaction_ids))

solution = model.cobra_model.optimize()
print(solution.objective_value)
```

For local analyses, load an SBML, JSON, YAML, or MAT model and then call pipeGEM workflows:

```python
import pipeGEM as pg
from pipeGEM.utils import load_model

cobra_model = load_model("your_model_path")
model = pg.Model(name_tag="model_name", model=cobra_model)

flux_analysis = model.do_flux_analysis("pFBA", solver="glpk")
flux_analysis.plot(
    rxn_ids=["rxn_a", "rxn_b"],
    file_name="pfba_flux.png",
)
```

## Multiple models

```python
import pipeGEM as pg
from pipeGEM.utils import load_model

model_a1 = load_model("your_model_path_1")
model_a2 = load_model("your_model_path_2")
model_b1 = load_model("your_model_path_3")
model_b2 = load_model("your_model_path_4")

group = pg.Group(
    {
        "group_a": {
            "model_a_dmso": model_a1,
            "model_a_metformin": model_a2,
        },
        "group_b": {
            "model_b_dmso": model_b1,
            "model_b_metformin": model_b2,
        },
    },
    name_tag="my_group",
    treatments={
        "model_a_dmso": "DMSO",
        "model_b_dmso": "DMSO",
        "model_a_metformin": "metformin",
        "model_b_metformin": "metformin",
    },
)

flux_analysis = group.do_flux_analysis("pFBA")
flux_analysis.plot(rxn_ids=["rxn_a", "rxn_b"])
```

## Context-specific models

```python
import numpy as np
import pipeGEM as pg
from pipeGEM.data import GeneData, synthesis
from pipeGEM.utils import load_model

model = pg.Model(name_tag="sample_0", model=load_model("your_model_path"))
dummy_data = synthesis.get_syn_gene_data(model, n_sample=3)

gene_data = GeneData(
    data=dummy_data["sample_0"],
    data_transform=lambda x: np.log2(x),
    absent_expression=-np.inf,
)

model.add_gene_data(
    name_or_prefix="sample_0",
    data=gene_data,
    or_operation="nanmax",
    threshold=-np.inf,
    absent_value=-np.inf,
)

result = model.integrate_gene_data(
    data_name="sample_0",
    integrator="GIMME",
    high_exp=5 * np.log10(2),
)

context_specific_model = result.result_model
```
