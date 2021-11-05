# PipeGEM v0.1.0
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/biodbs.svg)](https://pypi.python.org/pypi/biodbs/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
![ci](https://github.com/qwerty239qwe/biodbs/actions/workflows/ci.yml/badge.svg)
___
This is a package for visualizing and analyzing metabolic models. Also, it allows users to . 
The analysis functions in the package are based on cobrapy: 
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

model = load_model("your_model_path")
mod_cmpr = pg.Model(model)

# Print out model information
print(mod_cmpr)

# Do and plot pFBA result
mod_cmpr.do_flux_analysis("pFBA")
mod_cmpr.plot_flux_analysis("pFBA")
```


**multiple models**
```python
import pipeGEM as pg
from pipeGEM.utils import load_model

model_1 = load_model("your_model_path_1")
model_2 = load_model("your_model_path_2")
group = pg.Group({"model1": model_1, "model2": model_2})

# Do and plot pFBA result
group.do_flux_analysis("pFBA")
group.plot_flux_analysis("pFBA")
```

**Generate context-specific models**
```python
import pipeGEM as pg
```