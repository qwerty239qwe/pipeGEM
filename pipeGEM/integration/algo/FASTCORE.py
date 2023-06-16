from typing import Set, Union, List, Dict
from warnings import warn
from tqdm import tqdm

import cobra
import numpy as np

from pipeGEM.utils import get_rxn_set, flip_direction
from pipeGEM.analysis import find_sparse_mode, FASTCOREAnalysis, timing


@timing
def apply_FASTCORE(C: Union[List[str], Set[str]],
                   nonP: Union[List[str], Set[str]],
                   model: cobra.Model,
                   epsilon: float,
                   return_model: bool,
                   raise_err: bool = True,
                   scale_by_coef: bool = True) -> FASTCOREAnalysis:
    output_model = None
    if return_model:
        output_model = model.copy()
    if not isinstance(C, np.ndarray):
        C = np.array(list(C))
    if not isinstance(nonP, np.ndarray):
        nonP = np.array(list(nonP))
    all_rxns = get_rxn_set(model, dtype=np.array)
    irr_rxns = get_rxn_set(model, "irreversible", dtype=np.array)
    backward_rxns = get_rxn_set(model, "backward", dtype=np.array)

    with model:
        if len(backward_rxns) > 0:
            flip_direction(model, backward_rxns)

        flipped, singleton = False, False
        J = np.intersect1d(C, irr_rxns)
        P = np.setdiff1d(np.setdiff1d(all_rxns, C), nonP)
        singleJ = None
        A = np.array(find_sparse_mode(J, P, nonP, model, singleJ, epsilon))
        A.sort()
        invalid_part = np.setdiff1d(J, A)
        track_irrev = False
        if len(invalid_part) != 0:
            # track_irrev = True
            warn(f"Inconsistent irreversible core reactions (They should be included in A): Total: {invalid_part}")
        J = np.array(np.setdiff1d(C, A))  # reactions we want to add to the model
        with tqdm(total=len(J)) as pbar:
            n_j = len(J)
            while len(J) > 0:
                P = np.setdiff1d(P, A)
                supp = np.array(find_sparse_mode(J, P, nonP, model, singleJ, epsilon))
                A = np.union1d(A, supp)
                if len(np.intersect1d(J, A)) > 0:
                    J = np.setdiff1d(J, A)
                    pbar.update(n_j - len(J))
                    n_j = len(J)
                    flipped, singleton = False, False
                    singleJ = None

                    if track_irrev:
                        print(f"Irrev in J: {len(np.intersect1d(J, irr_rxns))}, J: {len(J)}")
                else:
                    if singleton and len(np.setdiff1d(J, irr_rxns)) != 0:
                        if singleJ is None:
                            Jrev = np.setdiff1d(J, irr_rxns)[0]
                            singleJ = np.array([Jrev])
                            flipped = False
                    else:
                        Jrev = np.setdiff1d(J, irr_rxns)
                    if len(Jrev) == 0 or flipped:  # If no reversible J or the model is flipped
                        if singleton:
                            if raise_err:
                                raise ValueError(f"Error: Global network is not consistent. \nLast rxn: {J}\n |J| = {len(J)}")
                            else:
                                to_remove = singleJ
                                warn(f"Error: Global network is not consistent. Removing core rxn: {to_remove}")
                                J = np.setdiff1d(J, to_remove)
                                n_j = len(J)
                                pbar.update(1)
                                flipped, singleton = False, False
                                singleJ = None
                        else:
                            flipped, singleton = False, True
                            if singleJ is None:
                                if len(np.setdiff1d(J, irr_rxns)) != 0:
                                    Jrev = np.array([np.setdiff1d(J, irr_rxns)[0]])
                                    singleJ = Jrev
                                elif len(np.intersect1d(J, irr_rxns)) != 0:  # this is for raise_err = False
                                    singleJ = np.array([np.intersect1d(J, irr_rxns)[0]])
                    else:
                        flip_direction(model, Jrev)
                        flipped = True
    rxns_to_remove = np.setdiff1d(all_rxns, A)
    if output_model is not None:
        output_model.remove_reactions(list(rxns_to_remove), remove_orphans=True)

    result = FASTCOREAnalysis(log={"epsilon": epsilon,})

    result.add_result(dict(result_model=output_model,
                           removed_rxn_ids=rxns_to_remove,
                           kept_rxn_ids=A))
    return result