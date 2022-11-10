# PipeGEM v0.1.0-alpha1
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/biodbs.svg)](https://pypi.python.org/pypi/pipeGEM/)
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
### How to use this package
**single model**
```python
import pipeGEM as pg
from pipeGEM.utils import load_model

model = load_model("your_model_path")  # cobra.Model
pmodel = pg.Model(model)

# Print out model information
print(pmodel)

# Do and plot pFBA result
flux_analysis = pmodel.do_flux_analysis("pFBA")
flux_analysis.plot()
```


**multiple models**
```python
import pipeGEM as pg
from pipeGEM.utils import load_model

model_1 = load_model("your_model_path_1")
model_2 = load_model("your_model_path_2")
group = pg.Group({"model1": model_1, "model2": model_2})

# Do and plot pFBA result
flux_analysis = group.do_flux_analysis("pFBA")
flux_analysis.plot()
```

**Generate context-specific models**
```python
import numpy as np
import pipeGEM as pg
from pipeGEM.utils import load_model
from pipeGEM.data import GeneData, synthesis

# initialize model
mod = pg.Model(load_model("your_model_path_1"))

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
gimme_result = mod.integrate_gene_data(data_name="sample_0", integrator="GIMME")
context_specific_gem = gimme_result.result_model

```