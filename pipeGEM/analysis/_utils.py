import numpy as np


def bh_adjust(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asarray(p, dtype=float)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]


def bonferroni_adjust(p):
    """Bonferroni p-value correction."""
    p = np.asarray(p, dtype=float)
    return np.minimum(p * len(p), 1.0)


def holm_adjust(p):
    """Holm-Bonferroni step-down p-value correction."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    for rank, idx in enumerate(order):
        adjusted[idx] = p[idx] * (n - rank)
    # Enforce monotonicity
    for i in range(1, n):
        idx = order[i]
        prev_idx = order[i - 1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    return np.minimum(adjusted, 1.0)


def adjust_p_values(p_values, method="bh"):
    """Apply p-value correction with selectable method.

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    method : str
        ``"bh"`` (Benjamini-Hochberg), ``"bonferroni"``, or ``"holm"``.

    Returns
    -------
    numpy.ndarray
        Adjusted p-values.
    """
    methods = {"bh": bh_adjust, "bonferroni": bonferroni_adjust, "holm": holm_adjust}
    if method not in methods:
        raise ValueError(f"Unknown correction method '{method}'. Choose from {list(methods)}.")
    return methods[method](p_values)
