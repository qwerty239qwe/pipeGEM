"""Data loading, preprocessing, and biological-data containers.

This submodule provides:

* **Data containers** — :class:`GeneData`, :class:`EnzymeData`,
  :class:`MediumData`, :class:`MetaboliteData` — that wrap biological
  measurements and align them with COBRA metabolic models.
* **Fetching utilities** — :func:`fetch_HPA_data`, :func:`list_models`,
  :func:`load_remote_model` — for downloading data and models from
  public databases (HPA, BiGG, Metabolic Atlas).
* **Preprocessing helpers** — :func:`translate_gene_id`,
  :func:`unify_score_column`, :func:`transform_HPA_data`,
  :func:`get_gene_id_map` — for gene-ID translation, score unification,
  and HPA data pivoting.
* **Synthetic data** — :func:`get_syn_gene_data` — for generating
  simulated gene-expression matrices for testing.
"""

from .data import GeneData, MediumData, EnzymeData, MetaboliteData, find_local_threshold
from .fetching import load_remote_model, list_models, fetch_HPA_data
from .synthesis import get_syn_gene_data
from .preprocessing import transform_HPA_data, unify_score_column, translate_gene_id, get_gene_id_map
from .medium_registry import MediumInfo, MediumCatalog


__all__ = ("GeneData",
           "EnzymeData",
           "MediumData",
           "MetaboliteData",
           "find_local_threshold",
           "load_remote_model",
           "list_models",
           "fetch_HPA_data",
           "get_syn_gene_data",
           "transform_HPA_data",
           "unify_score_column",
           "translate_gene_id",
           "get_gene_id_map",
           "MediumInfo",
           "MediumCatalog")