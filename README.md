# PipeGEM v0.2.0
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pipeGEM.svg)](https://pypi.python.org/pypi/pipeGEM/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![ci](https://github.com/qwerty239qwe/pipeGEM/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/qwerty239qwe/pipeGEM/graph/badge.svg?token=1BJAWO79OL)](https://codecov.io/gh/qwerty239qwe/pipeGEM)

___
PipeGEM is a Python package for analyzing and visualizing multiple genome-scale metabolic models (GEMs). It supports the integration of transcriptomic and proteomic data, metabolic task evaluation, and medium composition into GEMs. Flux analysis is powered by [cobrapy](https://cobrapy.readthedocs.io/en/latest/).

___
### Installation

**pip**
```bash
pip install pipegem
```

**uv**
```bash
uv add pipegem
```

**uv (development)**
```bash
git clone https://github.com/qwerty239qwe/pipeGEM.git
cd pipeGEM
uv sync
```

___
### Python API

**Single model**
```python
import pipeGEM as pg
from pipeGEM.utils import load_model

model = load_model("your_model_path")  # returns a cobra.Model
pmodel = pg.Model(name_tag="model_name", model=model)

print(pmodel)

flux_analysis = pmodel.do_flux_analysis("pFBA")
flux_analysis.plot(
    rxn_ids=['rxn_a', 'rxn_b'],
    file_name='pfba_flux.png'  # pass None to skip saving
)
```

**Multiple models**
```python
import pipeGEM as pg
from pipeGEM.utils import load_model

group = pg.Group(
    {
        "group_a": {
            "model_a_dmso": load_model("path_1"),
            "model_a_metformin": load_model("path_2"),
        },
        "group_b": {
            "model_b_dmso": load_model("path_3"),
            "model_b_metformin": load_model("path_4"),
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
flux_analysis.plot(rxn_ids=['rxn_a', 'rxn_b'])
```

**Context-specific models from omic data**

PipeGEM can reconstruct context-specific GEMs by integrating gene expression data. The example below uses GIMME, but a range of algorithms are available.

```python
import numpy as np
import pipeGEM as pg
from pipeGEM.utils import load_model
from pipeGEM.data import GeneData, synthesis

mod = pg.Model(name_tag="model_name", model=load_model("your_model_path"))

# Generate synthetic transcriptomic data for demonstration
dummy_data = synthesis.get_syn_gene_data(mod, n_sample=3)

gene_data = GeneData(
    data=dummy_data["sample_0"],
    data_transform=lambda x: np.log2(x),
    absent_expression=-np.inf,
)
mod.add_gene_data(
    name_or_prefix="sample_0",
    data=gene_data,
    or_operation="nanmax",  # alternative: "nansum"
    threshold=-np.inf,
    absent_value=-np.inf,
)

gimme_result = mod.integrate_gene_data(
    data_name="sample_0",
    integrator="GIMME",
    high_exp=5 * np.log10(2),
)
context_specific_gem = gimme_result.result_model
```

Supported integrators: GIMME, iMAT, FASTCORE, SWIFTCORE, MBA, mCADRE, CORDA, ftINIT, RIPTiDe, E-Flux, SPOT, rFASTCORMICS.

**Enzyme-constrained models (GECKO)**

After attaching enzyme kinetic data, call `integrate_enzyme_data` to produce an enzyme-constrained model.

```python
mod.add_enzyme_data(enzyme_data)  # EnzymeData object

ec_result = mod.integrate_enzyme_data(method="GECKOLight")  # or "GECKOFull"
ec_model = ec_result.result_model
```

**Logging**

PipeGEM is silent by default. To enable progress output, adjust the log level before running analyses:

```python
import logging
import pipeGEM as pg

pg.set_log_level(logging.INFO)  # show progress messages
pg.enable_verbose()             # enable DEBUG output with a StreamHandler to stderr
```

___
### CLI

PipeGEM provides a command-line interface organized around subcommands. To see all available options:

```bash
pipeGEM --version
pipeGEM --help
```

**Step 1 — Generate template config files**

```bash
pipeGEM template -p integration -o ./configs
```

This creates a `configs/` directory containing TOML templates for each required config file.

**Step 2 — Edit the configs**

Fill in your model paths, data paths, and algorithm parameters in the generated TOML files.

**Step 3 — Run a pipeline**

Add `--dry-run` to any command to validate the configs and preview the planned actions without executing them.

```bash
# Process a model
pipeGEM process -t configs/model_conf.toml

# Find expression thresholds
pipeGEM threshold -g configs/gene_data_conf.toml -r configs/threshold_conf.toml

# Full context-specific model reconstruction
pipeGEM integrate \
    -g configs/gene_data_conf.toml \
    -t configs/model_conf.toml \
    -r configs/threshold_conf.toml \
    -m configs/mapping_conf.toml \
    -i configs/integration_conf.toml

# Flux analysis
pipeGEM flux -f configs/flux_conf.toml -t configs/model_conf.toml

# Compare models across conditions
pipeGEM compare -c configs/comparison_conf.toml
```

> **Note:** The legacy `-n <pipeline>` style is still accepted for backward compatibility but is deprecated. Please migrate to the subcommand style shown above.

___
### What's new in 0.2.0

- **CLI subcommands** — the flat `-n <pipeline>` interface has been replaced with proper subcommands (`integrate`, `process`, `threshold`, `flux`, `compare`, `template`). The old style still works but emits a deprecation warning.
- **`--dry-run` flag** — available on all subcommands; validates configs and prints the planned actions without running the pipeline.
- **`integrate_enzyme_data()`** — now fully implemented. Accepts `method="GECKOLight"` (default) or `"GECKOFull"`.
- **Silent by default** — all internal `print` calls have been replaced with structured loggers under the `pipeGEM` namespace. Use `pg.set_log_level` or `pg.enable_verbose` to opt in to output.
- **Bug fixes:**
  - `Model.rename()` silently swallowed a `TypeError` when given a non-string argument — it now raises correctly.
  - `PairwiseTester` always selected non-parametric methods regardless of the normality test result.
  - `data.preprocessing`: column drops were incorrectly targeting rows; a row-wise `apply` was missing `axis=1`; `na_action=""` was invalid and replaced with `na_action=None`.
  - `fetch_HPA_data` updated to use the current `biodbs` API (`hpa_search`).
