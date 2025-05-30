import cobra
import numpy as np
from typing import Union

import cobra
from tqdm import tqdm
from time import time


class RxnMapper:
    def __init__(self,
                 data,
                 model: Union[cobra.Model],
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
            Operation to apply to inner gene-rule-reaction scores, by default "nanmin".
        or_operation : str, optional
            Operation to apply to outer gene-rule-reaction scores, by default "nanmax".
        plus_operation : str, optional
            Operation to apply to reduced reaction scores, by default "nansum".
        **kwargs
            Additional keyword arguments to pass to `_map_to_rxns`.
        """
        self.genes = data.genes
        self.gene_data = data.gene_data
        self.missing_value = missing_value  # if the gene is not shown in the given data
        self.rxn_scores = self._map_to_rxns(model,
                                            threshold=threshold,
                                            absent_value=absent_value,
                                            and_operation=and_operation,
                                            or_operation=or_operation,
                                            plus_operation=plus_operation,
                                            **kwargs)

    def _inner_grr_helper(self, grr_list) -> list:
        """
        Helper function to compute inner gene-rule-reaction scores.

        Parameters
        ----------
        grr_list : str
            Gene-rule-reaction string to compute scores for.

        Returns
        -------
        list
            List of scores for the given gene-rule-reaction string.
        """
        inner_grr_scores = []
        for g in grr_list.split('and'):
            if g == "" or not g in self.genes:
                inner_grr_scores.append(self.missing_value)
            else:
                inner_grr_scores.append(self.gene_data[g])
        return inner_grr_scores

    def _outer_grr_helper(self,
                          inner_grr_scores,
                          operation="nanmin") -> float:
        """
        Helper function to compute outer gene-rule-reaction scores.

        Parameters
        ----------
        inner_grr_scores : list
            List of inner gene-rule-reaction scores.
        operation : str, optional
            Operation to apply to the inner scores, by default "nanmin".

        Returns
        -------
        float
            Score for the given outer gene-rule-reaction string.
        """
        if len(inner_grr_scores) == 0:
            return self.missing_value
        if all([not np.isfinite(i) for i in inner_grr_scores]):
            return self.missing_value
        return getattr(np, operation)(inner_grr_scores)

    def _map_to_rxns(self,
                     model,
                     absent_value=0,
                     threshold=0.,
                     and_operation="nanmin",
                     or_operation="nanmax",
                     plus_operation="nansum",
                     gene_ids = None):
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
            NumPy operation to use for combining OR conditions. Default is "nansum".
        gene_ids : list, optional
            Subset of gene IDs to map to reactions.

        Returns
        -------
        dict
            Dictionary of reaction IDs and their scores.
        """
        start_time = time()

        grrs = {r.id: r.gene_reaction_rule.replace(' ', '').replace('(', '').replace(')', '')
                for r in model.reactions} if gene_ids is None else \
               {r.id: r.gene_reaction_rule.replace(' ', '').replace('(', '').replace(')', '')
                for r in model.reactions if len(set([g.id for g in r.genes]) & set(gene_ids)) > 0}

        rxn_score = {}
        for rxn_id, grr in tqdm(grrs.items()):
            if len(grr) == 0:
                rxn_score[rxn_id] = self.missing_value
                continue
            plus_grr_list = []
            for grr_i in grr.split("plus"):
                outer_grr_list = []
                for gl in grr_i.split("or"):
                    inner_grr_scores = self._inner_grr_helper(gl)
                    outer_grr_list.append(self._outer_grr_helper(inner_grr_scores, and_operation))
                plus_grr_list.append(self._outer_grr_helper(outer_grr_list, or_operation))
            if all([not np.isfinite(i) for i in plus_grr_list]):
                rxn_score[rxn_id] = self.missing_value
            else:
                rxn_score[rxn_id] = getattr(np, plus_operation)(plus_grr_list)
            rxn_score[rxn_id] = rxn_score[rxn_id] if np.isfinite(rxn_score[rxn_id]) else self.missing_value

        for k, v in rxn_score.items():
            if v <= threshold:
                rxn_score[k] = absent_value
        print(f"Finished mapping in {time() - start_time} seconds.")
        return rxn_score

    def partial_map(self, model, new_data, gene_ids, **kwargs):
        new_rxn_score = self._map_to_rxns(model=model, gene_ids=gene_ids, **kwargs)
        self.rxn_scores.update(new_rxn_score)
