from typing import Dict, List, Callable, Union

import cobra
import numpy as np

from pipeGEM.utils import select_rxns_from_model
from pipeGEM.utils.transform import exp_x
from pipeGEM.analysis import EFluxAnalysis


def apply_EFlux(model: cobra.Model,
                rxn_expr_score: Dict[str, float],
                max_ub: float = 1000,
                min_lb: float = .03,
                ignore: Union[str, List[str], None] = None,
                transform: Callable = exp_x):
    """
    Parameters
    ----------
    model: cobra.Model
        A cobra model to be analyzed
    rxn_expr_score: dict
        A dict with rxn ids as keys and expression as values
    max_ub: float or int
        max upper bound that will apply to the max expressed reaction
    min_lb: float or int
        min lower bound that will apply to the min expressed reaction
    ignore: str or list of str
        rxn_ids / subsystems to be ignored
    plot_exp: bool
        To plot expression vs constraints scatter plot or not
    transform: callable
        User determined transformation function

    Returns
    -------
    """
    assert max_ub > 0, "max_ub should be a positive number"
    assert min_lb > 0, "min_lb should be a positive number"
    assert max_ub - min_lb > 0, "max_ub should be larger than min_lb"
    if ignore:
        ignore_rxn_ids = select_rxns_from_model(model, ignore, return_id=True)
        print(f"Ignoring {ignore_rxn_ids}")
    else:
        ignore_rxn_ids = []
    exps = [v for _, v in rxn_expr_score.items() if not np.isnan(v)]
    max_exp = max(exps) if len(exps) != 0 else max_ub
    min_exp = min(exps) if len(exps) != 0 else min_lb
    print(f"Max expression: {max_exp} | Min expression: {min_exp}")
    assert max_exp > 0, "max_exp should be a positive number, all expression values might be zeros"
    exchanges = model.exchanges
    trans_min_exp = transform(min_exp)
    denominator = transform(max_exp) - trans_min_exp
    trans_rxn_exp_dict = {k: min_lb + ((max_ub - min_lb) * (transform(v) - trans_min_exp) / denominator)
                          if not np.isnan(v) else v
                          for k, v in rxn_expr_score.items()}
    r_bounds_dict = {}
    for r in model.reactions:
        if r not in exchanges and r.id not in ignore_rxn_ids:
            if not np.isnan(trans_rxn_exp_dict[r.id]) and (-trans_rxn_exp_dict[r.id] > r.lower_bound):
                r.lower_bound = -trans_rxn_exp_dict[r.id]
            if not np.isnan(trans_rxn_exp_dict[r.id]) and (trans_rxn_exp_dict[r.id] < r.upper_bound):
                r.upper_bound = trans_rxn_exp_dict[r.id]
        r_bounds_dict[r.id] = r.bounds

    result = EFluxAnalysis(log={"name": model.name, "max_ub": max_ub, "min_lb": min_lb, "ignored_rxns": ignore})
    result.add_result(rxn_bounds=r_bounds_dict, rxn_scores=rxn_expr_score)
    return result
