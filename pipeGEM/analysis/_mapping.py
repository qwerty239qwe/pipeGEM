from ast import And, BoolOp, Name, Or
from math import isfinite
from time import time

import cobra
import numpy as np
from tqdm import tqdm

from pipeGEM._logging import get_logger
from ._reducing import MergedReaction

logger = get_logger(__name__)


class RxnMapper:
    def __init__(self,
                 data,
                 model: cobra.Model,
                 threshold=0,
                 absent_value=0,
                 missing_value=np.nan,
                 and_operation="nanmin",
                 or_operation="nanmax",
                 plus_operation="nansum",  # for reduced reactions
                 **kwargs
                 ):
        """
        Parameters
        ----------
        data : GeneData
            GeneData containing gene data.
        model : cobra.Model or pipeGEM.Model
            Model containing model reactions.
        threshold : float, optional
            Reaction score threshold, by default 0.
        absent_value : int, optional
            Value to use when reaction is absent, by default 0.
        missing_value : float, optional
            Value to use when gene is missing, by default np.nan.
        and_operation : str, optional
            NumPy reduction for AND nodes in the GPR tree, by default "nanmin".
        or_operation : str, optional
            NumPy reduction for OR nodes in the GPR tree, by default "nanmax".
        plus_operation : str, optional
            Operation to apply to reduced reaction scores, by default "nansum".
        **kwargs
            Additional keyword arguments to pass to `_map_to_rxns`.
        """
        self.genes = data.genes
        self.gene_data = data.gene_data
        self.missing_value = missing_value  # if the gene is not shown in the given data
        self._mapping_options = dict(threshold=threshold, absent_value=absent_value,
                                     and_operation=and_operation, or_operation=or_operation,
                                     plus_operation=plus_operation, **kwargs)
        self.rxn_scores = self._map_to_rxns(model, **self._mapping_options)

    def _reduce_scores(self, scores, operation):
        if not any(isfinite(score) for score in scores):
            return self.missing_value
        if len(scores) == 1 and operation is np.nansum:
            return scores[0]
        return operation(scores)

    def _map_to_rxns(self,
                     model,
                     absent_value=0,
                     threshold=0.,
                     and_operation="nanmin",
                     or_operation="nanmax",
                     plus_operation="nansum",
                     gene_ids=None):
        """
        Map genes to reactions based on a given metabolic model.

        Parameters
        ----------
        model : cobra.Model
            Metabolic model to map genes to reactions.
        absent_value : float, optional
            Value to use for reactions that don't meet the threshold. Default is 0.
        threshold : float, optional
            Threshold score below which reactions will be set to absent_value. Default is 0.
        and_operation : str, optional
            NumPy operation to use for AND conditions. Default is "nanmin".
        or_operation : str, optional
            NumPy operation to use for OR conditions. Default is "nanmax".
        plus_operation : str, optional
            NumPy operation for combining merged reaction components. Default is "nansum".
        gene_ids : list, optional
            Subset of gene IDs to map to reactions.

        Returns
        -------
        dict
            Dictionary of reaction IDs and their scores.
        """
        start_time = time()
        reducers = {And: getattr(np, and_operation), Or: getattr(np, or_operation)}
        plus_reducer = getattr(np, plus_operation)
        requested_genes = None if gene_ids is None else set(gene_ids)
        gene_data = self.gene_data

        def evaluate(node):
            if isinstance(node, Name):
                return gene_data.get(node.id, self.missing_value)
            if isinstance(node, BoolOp):
                return self._reduce_scores([evaluate(child) for child in node.values],
                                           reducers[type(node.op)])
            raise TypeError(f"Unsupported GPR node: {type(node).__name__}")

        rxn_score = {}
        for reaction in tqdm(model.reactions):
            if requested_genes is not None and not any(
                    gene.id in requested_genes for gene in reaction.genes):
                continue
            if isinstance(reaction, MergedReaction):
                scores = [evaluate(component.gpr.body) for component in reaction.merged_rxns
                          if component.gpr.body is not None]
            elif reaction.gpr.body is None:
                scores = []
            else:
                scores = [evaluate(reaction.gpr.body)]
            score = self._reduce_scores(scores, plus_reducer)
            score = score if isfinite(score) else self.missing_value
            rxn_score[reaction.id] = absent_value if score <= threshold else score
        logger.info("Finished mapping in %s seconds.", time() - start_time)
        return rxn_score

    def partial_map(self, model, new_data, gene_ids, **kwargs):
        """Refresh affected reactions from a complete updated GeneData dataset.

        ``gene_ids`` identifies changed genes. Other reaction scores are retained;
        constructor mapping options apply unless overridden for this update.
        """
        options = {**self._mapping_options, **kwargs, "gene_ids": gene_ids}
        self.genes = new_data.genes
        self.gene_data = new_data.gene_data
        new_rxn_score = self._map_to_rxns(model=model, **options)
        self.rxn_scores.update(new_rxn_score)
