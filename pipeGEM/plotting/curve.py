import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_rFastCormic_thresholds(x,
                                y,
                                exp_th,
                                nonexp_th,
                                right_c = None,
                                left_c = None):
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


def plot_percentile_thresholds(data, exp_th, figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.histplot(data=data, ax=ax, kde=True)
    y_lims = ax.get_ylim()
    ax.axvline(x=exp_th, ymax=y_lims[1], label="Expression threshold")
    ax.set_ylim(*y_lims)
    ax.legend()
    return {"g": fig}