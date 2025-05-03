from typing import Set, Union, List, Dict
from warnings import warn
from tqdm import tqdm

import cobra
import numpy as np
import pandas as pd

from pipeGEM.utils import get_rxn_set, flip_direction
from pipeGEM.analysis import find_sparse_mode, FASTCOREAnalysis, timing, measure_efficacy


@timing
def apply_FASTCORE(C: Union[List[str], Set[str]],
                   nonP: Union[List[str], Set[str]],
                   model: cobra.Model,
                   epsilon: float,
                   return_model: bool,
                   copy_model: bool = True,
                   raise_err: bool = True,
                   rxn_scaling_coefs: dict = None,
                   calc_efficacy: bool = True) -> FASTCOREAnalysis:
    """Apply the FASTCORE algorithm to extract a flux-consistent subnetwork.

    FASTCORE identifies a minimal set of reactions from a given metabolic model
    that includes a defined set of core reactions (C) while ensuring flux
    consistency. Non-penalty reactions (nonP) can be included without affecting
    the objective function during the sparse mode search.

    Parameters
    ----------
    C : list[str] or set[str]
        Core reaction IDs that must be included and carry flux.
    nonP : list[str] or set[str]
        Non-penalty reaction IDs. Included if needed but not prioritized.
    model : cobra.Model
        Input genome-scale metabolic model.
    epsilon : float
        Tolerance threshold for flux consistency checks. Flux values below
        this are considered zero. Adjusted per reaction if `rxn_scaling_coefs`
        is provided.
    return_model : bool
        If True, return the extracted subnetwork as a cobra.Model object.
    copy_model : bool, optional
        If True (default), operate on a copy of the input model.
        If False, modify the input model directly.
    raise_err : bool, optional
        If True (default), raise ValueError on inconsistency.
        If False, warn and potentially remove problematic core reactions.
    rxn_scaling_coefs : dict, optional
        Mapping of reaction IDs to scaling coefficients to adjust `epsilon`
        per reaction. Defaults to None.
    calc_efficacy : bool, optional
        If True (default), calculate efficacy metrics.

    Returns
    -------
    FASTCOREAnalysis
        An object containing the results:
        - result_model (cobra.Model, optional): Extracted subnetwork (if `return_model` is True).
        - kept_rxn_ids (np.ndarray): Reaction IDs included in the subnetwork.
        - removed_rxn_ids (np.ndarray): Reaction IDs excluded from the subnetwork.
        - algo_efficacy (dict, optional): Efficacy metrics (if `calc_efficacy` is True).
        - log (dict): Algorithm parameters like epsilon.

    Raises
    ------
    ValueError
        If `raise_err` is True and an inconsistency prevents including all core reactions.

    Notes
    -----
    Based on the algorithm described in: Vlassis, N., Pacheco, M. P., & Sauter, T. (2014).
    Fast reconstruction of compact context-specific metabolic networks. PLoS computational biology, 10(1), e1003424.
    """
    output_model = None
    if return_model:
        output_model = model.copy() if copy_model else model
    if not isinstance(C, np.ndarray):
        C = np.array(list(C))
    if not isinstance(nonP, np.ndarray):
        nonP = np.array(list(nonP))
    all_rxns = get_rxn_set(model, dtype=np.array)
    irr_rxns = get_rxn_set(model, "irreversible", dtype=np.array)
    backward_rxns = get_rxn_set(model, "backward", dtype=np.array)
    if rxn_scaling_coefs is not None:
        tol = pd.Series({k: epsilon / v
                        for k, v in rxn_scaling_coefs.items()}).sort_index()
    else:
        tol = epsilon

    with model:
        if len(backward_rxns) > 0:
            flip_direction(model, backward_rxns)
        flipped, singleton = False, False
        J = np.intersect1d(C, irr_rxns)
        P = np.setdiff1d(np.setdiff1d(all_rxns, C), nonP)
        singleJ = None
        A = np.array(find_sparse_mode(J, P, nonP, model, singleJ, tol))
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
                supp = np.array(find_sparse_mode(J, P, nonP, model, singleJ, tol))
                A = np.union1d(A, supp)
                if len(np.intersect1d(J, A)) > 0:
                    J = np.setdiff1d(J, A)
                    pbar.update(n_j - len(J))
                    n_j = len(J)
                    flipped, singleton = False, False
                    singleJ = None

                    if track_irrev:
                        tqdm.write(f"Irrev in J: {len(np.intersect1d(J, irr_rxns))}, J: {len(J)}")
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
                                tqdm.write(f"Warning: Global network is not consistent. Removing core rxn: {to_remove}")
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

    result = FASTCOREAnalysis(log={"epsilon": tol,})
    algo_efficacy = None
    if calc_efficacy:
        algo_efficacy = measure_efficacy(kept_rxn_ids=list(A),
                                         removed_rxn_ids=rxns_to_remove,
                                         core_rxn_ids=list(C),
                                         non_core_rxn_ids=list(P))

    result.add_result(dict(result_model=output_model,
                           removed_rxn_ids=rxns_to_remove,
                           kept_rxn_ids=A,
                           algo_efficacy=algo_efficacy))

    return result
