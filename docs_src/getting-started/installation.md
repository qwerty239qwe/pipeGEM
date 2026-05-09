# Installation

pipeGEM supports Python 3.10 and newer.

## Install from PyPI

```bash
pip install pipegem
```

## Install from source

```bash
git clone https://github.com/qwerty239qwe/pipeGEM.git
cd pipeGEM
uv sync
```

## Optional extras

Install documentation dependencies when building these docs locally:

```bash
uv run --extra doc mkdocs build --strict -d ./docs
```

DLKcat support requires additional machine-learning and cheminformatics dependencies:

```bash
uv sync --extra dlkcat
```

## Check the installation

```bash
pipeGEM --version
pipeGEM --help
```

In Python:

```python
import pipeGEM as pg

print(pg.__version__)
```
