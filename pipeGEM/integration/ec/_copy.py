"""Copy helpers for enzyme-constrained model builders."""

from copy import deepcopy


def copy_cobra_model(model):
    """Return an independent copy of a COBRA model.

    COBRA models contain cyclic references among models, reactions,
    metabolites, and genes.  Python 3.14's generic ``deepcopy`` can recurse
    through those cycles deeply enough to hit ``RecursionError``.  COBRA's
    own ``Model.copy()`` is designed for this object graph and is also faster.
    """
    if hasattr(model, "copy"):
        return model.copy()
    return deepcopy(model)
