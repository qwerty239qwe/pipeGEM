"""KEGG pathway integration for metabolic models."""
import warnings
from typing import Dict, List, Optional

import requests
import pandas as pd

from pipeGEM._logging import get_logger

logger = get_logger(__name__)

_KEGG_BASE = "https://rest.kegg.jp"


class KEGGPathwayMapper:
    """Fetch and map KEGG pathway definitions to model reactions.

    Parameters
    ----------
    organism : str
        KEGG organism code (e.g. ``"hsa"`` for human, ``"eco"`` for
        *E. coli*).
    """

    def __init__(self, organism: str = "hsa"):
        self.organism = organism
        self._pathway_cache: Dict[str, List[str]] = {}
        self._pathway_names: Dict[str, str] = {}

    def fetch_pathways(self) -> Dict[str, str]:
        """Fetch all metabolic pathways for the organism from KEGG.

        Returns
        -------
        dict
            Mapping of pathway IDs to pathway names.
        """
        url = f"{_KEGG_BASE}/list/pathway/{self.organism}"
        logger.info("Fetching pathways from %s", url)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Failed to fetch pathways: %s", exc)
            return {}

        pathways = {}
        for line in resp.text.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                pid = parts[0].strip()
                pname = parts[1].strip()
                pathways[pid] = pname
        self._pathway_names = pathways
        logger.info("Fetched %d pathways for organism '%s'.", len(pathways), self.organism)
        return pathways

    def fetch_pathway_genes(self, pathway_id: str) -> List[str]:
        """Fetch gene IDs for a specific pathway.

        Parameters
        ----------
        pathway_id : str
            KEGG pathway ID (e.g. ``"hsa00010"``).

        Returns
        -------
        list of str
            Gene IDs associated with the pathway.
        """
        if pathway_id in self._pathway_cache:
            return self._pathway_cache[pathway_id]

        url = f"{_KEGG_BASE}/link/genes/{pathway_id}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Failed to fetch genes for %s: %s", pathway_id, exc)
            return []

        genes = []
        for line in resp.text.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                gene_id = parts[1].strip()
                # Remove organism prefix if present
                if ":" in gene_id:
                    gene_id = gene_id.split(":")[1]
                genes.append(gene_id)

        self._pathway_cache[pathway_id] = genes
        return genes

    def map_to_model(
        self,
        model,
        pathways: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[str]]:
        """Map KEGG pathways to model reaction IDs via gene associations.

        Parameters
        ----------
        model : cobra.Model
            The metabolic model.
        pathways : dict, optional
            Pathway dict from :meth:`fetch_pathways`.  If ``None``,
            pathways are fetched automatically.

        Returns
        -------
        dict
            Mapping of pathway IDs to lists of reaction IDs present in
            the model.
        """
        if pathways is None:
            pathways = self.fetch_pathways()

        # Build gene -> reactions lookup
        gene_to_rxns: Dict[str, List[str]] = {}
        for rxn in model.reactions:
            for gene in rxn.genes:
                gene_to_rxns.setdefault(gene.id, []).append(rxn.id)

        pathway_reactions: Dict[str, List[str]] = {}
        for pid in pathways:
            genes = self.fetch_pathway_genes(pid)
            rxn_ids = set()
            for g in genes:
                if g in gene_to_rxns:
                    rxn_ids.update(gene_to_rxns[g])
            if rxn_ids:
                pathway_reactions[pid] = sorted(rxn_ids)

        logger.info(
            "Mapped %d/%d pathways to model reactions.",
            len(pathway_reactions), len(pathways),
        )
        return pathway_reactions
