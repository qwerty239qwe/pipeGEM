from typing import Dict

from optlang.symbolics import Zero
import cobra
import numpy as np
import pandas as pd
from cobra.util import add_absolute_expression

from pipeGEM.utils import make_irrev_rxn, merge_irrevs_in_df
from pipeGEM.integration.constraints import register


@register
def LAD(model: cobra.Model,
        rxn_expr_score: Dict[str, float],
        low_exp: float,
        high_exp: float,
        obj_frac: float = 0.8,
        get_details: bool = True):
    """
    GLF implementation

    Parameters
    ----------
    model: cobra.Model
        The analyzed model, should has set RMF
    rxn_expr_score: Dict[str, float]
        A dict with rxn_ids as keys and expression values as values
    low_exp: float
        Expression value lower than this value is treated as 0
    high_exp: float
        Expression value higher than this value is treated as high_exp
    obj_frac: float

    get_details: bool

    Returns
    -------
    None
    """
    NotImplementedError()
