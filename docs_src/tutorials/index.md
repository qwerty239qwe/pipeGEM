# Tutorials

The tutorial notebooks from the Sphinx documentation are preserved as notebooks and rendered without execution during documentation builds.

## Notebook sequence

1. [Basic operations](1_basic.ipynb): model loading, wrapping, inspection, and basic operations.
2. [Data fetching and processing](2_data.ipynb): gene-data fetching, preprocessing, and mapping.
3. [Model testing](3_model_testing.ipynb): model consistency and metabolic task testing.
4. [Model extraction methods](4_MEM.ipynb): context-specific model generation from gene-expression data.
5. [Flux simulation and visualization](5_flux_simulation.ipynb): flux analysis and plotting.

Notebook execution is disabled for docs builds so that CI remains stable across solver and remote-data environments.
