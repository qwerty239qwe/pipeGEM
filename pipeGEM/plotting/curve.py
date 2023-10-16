import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from ._utils import handle_colors


def plot_rFastCormic_thresholds(x: np.ndarray,
                                y: np.ndarray,
                                exp_th: float,
                                nonexp_th: float,
                                right_c: Optional[np.ndarray] = None,
                                left_c: Optional[np.ndarray] = None) -> dict:
    """
    Plots a gene expression distribution along with fitted expressed and non-expressed distributions, and gene
    expression thresholds.

    Parameters
    ----------
    x : numpy.ndarray
        A numpy array containing the values of the x-axis.
    y : numpy.ndarray
        A numpy array containing the values of the y-axis.
    exp_th : float
        The threshold for expressed genes.
    nonexp_th : float
        The threshold for non-expressed genes.
    right_c : Optional[numpy.ndarray], optional
        A numpy array containing the values of the fitted expressed distribution, by default None.
    left_c : Optional[numpy.ndarray], optional
        A numpy array containing the values of the fitted non-expressed distribution, by default None.

    Returns
    -------
    dict
        A dictionary containing the plot figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, label="Data")
    if right_c is not None:
        ax.plot(x, right_c, label="Fitted expressed distribution")
    if left_c is not None:
        ax.plot(x, left_c, label="Fitted non-expressed distribution")

    ax.plot([nonexp_th, nonexp_th], [0, np.max(y)], label="Nonexpressed gene threshold")
    ax.plot([exp_th, exp_th], [0, np.max(y)], label="Expressed gene threshold")
    ax.legend()
    return {"g": fig}


def plot_percentile_thresholds(data: Union[pd.DataFrame, pd.Series],
                               exp_th: Union[float, pd.Series],
                               figsize: Tuple[float, float] = (8, 6),
                               palatte: str = "deep",
                               **kwargs) -> dict:
    """
    Plots a histogram of the input data and a vertical line indicating the expression threshold.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        The data to plot in the histogram.
    exp_th : float, pd.Series
        The threshold for expressed genes.
    figsize : tuple, optional
        A tuple containing the width and height of the figure in inches, by default (8, 6).
    palatte: str
        Palatte used to draw the distribution and thresholds.

    Returns
    -------
    dict
        A dictionary containing the plot figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    n_needed_colors = 1 if isinstance(exp_th, float) else len(exp_th)

    colors = handle_colors(palette=palatte,
                           n_colors_used=1+n_needed_colors)
    ax = sns.histplot(data=data,
                      ax=ax,
                      kde=True,
                      color=colors[0],
                      **kwargs)
    y_lims = ax.get_ylim()
    ax.plot([], label="Data", color=colors[0])
    if isinstance(exp_th, float):
        ax.axvline(x=exp_th, ymax=y_lims[1], color=colors[1], label="Expression threshold")
    elif isinstance(exp_th, pd.Series) or isinstance(exp_th, dict):
        for i, (idx, e) in enumerate(exp_th.items()):
            ax.axvline(x=e,
                       ymax=y_lims[1],
                       label=f"{idx}",
                       color=colors[i+1])
    elif isinstance(exp_th, list) or isinstance(exp_th, np.ndarray):
        for i, e in enumerate(exp_th):
            if i == 0:
                ax.axvline(x=e, ymax=y_lims[1], color=colors[i+1], label="Expression threshold")
            else:
                ax.axvline(x=e, ymax=y_lims[1], color=colors[i+1])
    ax.set_ylim(*y_lims)
    ax.legend()
    return {"g": fig}