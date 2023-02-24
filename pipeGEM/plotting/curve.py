import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union


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
                               exp_th: float,
                               figsize: Tuple[float, float] = (8, 6)) -> dict:
    """
    Plots a histogram of the input data and a vertical line indicating the expression threshold.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        The data to plot in the histogram.
    exp_th : float
        The threshold for expressed genes.
    figsize : tuple, optional
        A tuple containing the width and height of the figure in inches, by default (8, 6).

    Returns
    -------
    dict
        A dictionary containing the plot figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.histplot(data=data, ax=ax, kde=True)
    y_lims = ax.get_ylim()
    ax.axvline(x=exp_th, ymax=y_lims[1], label="Expression threshold")
    ax.set_ylim(*y_lims)
    ax.legend()
    return {"g": fig}