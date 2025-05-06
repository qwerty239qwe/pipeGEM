# PipeGEM v0.1.0
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pipeGEM.svg)](https://pypi.python.org/pypi/pipeGEM/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![ci](https://github.com/qwerty239qwe/pipeGEM/actions/workflows/ci.yml/badge.svg)
___
This is a package for visualizing and analyzing multiple metabolic models. 
It also allow users to integrate omic data, metabolic tasks, and medium data with GEMs. 

The flux analysis functions in the package are based on cobrapy: 
https://cobrapy.readthedocs.io/en/latest/
___
### How to get PipeGEM
To install directly from PyPI:
<br>
`pip install pipegem`
___
### How to use this package (Python API)
**single model**
```python
import pipeGEM as pg
from pipeGEM.utils import load_model

model = load_model("your_model_path")  # cobra.Model
pmodel = pg.Model(name_tag="model_name", 
                  model=model)

# Print out model information
print(pmodel)

# Do and plot pFBA result
flux_analysis = pmodel.do_flux_analysis("pFBA")
flux_analysis.plot(
    rxn_ids=['rxn_a', 'rxn_b'],
    file_name='pfba_flux.png'  # can be None if you don't want to save the figure
    )
```


**multiple models**
```python
import pipeGEM as pg
from pipeGEM.utils import load_model

model_a1 = load_model("your_model_path_1")
model_a2 = load_model("your_model_path_2")

model_b1 = load_model("your_model_path_3")
model_b2 = load_model("your_model_path_4")

group = pg.Group({
        "group_a": {
            "model_a_dmso": model_a1, 
            "model_a_metformin": model_a2
        },
        "group_b": {
            "model_b_dmso": model_b1, 
            "model_b_metformin": model_b2
        }
    }, 
    name_tag="my_group", 
    treatments={"model_a_dmso": "DMSO", 
                "model_b_dmso": "DMSO",
                "model_a_metformin": "metformin", 
                "model_b_metformin": "metformin"}
)

# Do and plot pFBA result
flux_analysis = group.do_flux_analysis("pFBA")
flux_analysis.plot(rxn_ids=['rxn_a', 'rxn_b'])
```

**Generate context-specific models**
```python
import numpy as np
import pipeGEM as pg
from pipeGEM.utils import load_model
from pipeGEM.data import GeneData, synthesis

# initialize model
mod = pg.Model(name_tag="model_name", 
               model=load_model("your_model_path_1"))

# create dummy transcriptomic data
dummy_data = synthesis.get_syn_gene_data(mod, n_sample=3)

# calculate reaction activity score
gene_data = GeneData(data=dummy_data["sample_0"], # pd.Series or a dict
                     data_transform=lambda x: np.log2(x), # callable
                     absent_expression=-np.inf) # value
mod.add_gene_data(name_or_prefix="sample_0",  # name of the data
                  data=gene_data, 
                  or_operation="nanmax",  # alternative: nansum
                  threshold=-np.inf, 
                  absent_value=-np.inf)

# apply GIMME algorithm on the model
gimme_result = mod.integrate_gene_data(data_name="sample_0", integrator="GIMME", high_exp=5*np.log10(2))
context_specific_gem = gimme_result.result_model

```

___

### Command-Line Interface (CLI) Quick Start

PipeGEM also provides a command-line interface for running predefined pipelines using configuration files.

1.  **Generate Template Configurations:**
    Start by generating template TOML configuration files for a specific pipeline (e.g., `integration`). Replace `integration` with the desired pipeline name if needed.

    ```bash
    python -m pipeGEM -n template -p integration -o ./configs
    ```
    This will create a `configs` directory (if it doesn't exist) containing template `.toml` files like `gene_data_conf.toml`, `model_conf.toml`, etc.

2.  **Modify Configurations (Optional):**
    Edit the generated `.toml` files in the `configs` directory to specify your input file paths, parameters, and desired settings. For example, in `model_conf.toml`, you might specify the path to your metabolic model file.

3.  **Run a Pipeline:**
    Execute a pipeline using the configuration files. For example, to run the model processing pipeline using the configuration in `configs/model_conf.toml`:

    ```bash
    python -m pipeGEM -n model_processing -t configs/model_conf.toml
    ```

    Or, to run the full integration pipeline:

    ```bash
    python -m pipeGEM -n integration \
        -g configs/gene_data_conf.toml \
        -t configs/model_conf.toml \
        -r configs/threshold_conf.toml \
        -m configs/mapping_conf.toml \
        -i configs/integration_conf.toml
    ```

    Refer to the generated template files and the specific pipeline documentation for details on required configurations.