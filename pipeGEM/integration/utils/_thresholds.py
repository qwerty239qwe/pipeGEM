import warnings

from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


__all__ = ["get_rfastcormics_thresholds", "get_PROM_threshold"]


def gaussian(x, amp, cen, wid):
    return amp * np.exp(-np.power(x-cen, 2) / wid)


def bimodal(x, A1, mu1, wid1, A2, mu2, wid2):
    return gaussian(x, A1, mu1, wid1)+gaussian(x, A2, mu2, wid2)


def get_second_deriv(y, dx):
    return np.gradient(np.gradient(y, dx), dx)


def find_canyons(x, y, n=2):
    dx = x[1] - x[0]
    y = get_second_deriv(y, dx)

    prev = None
    cans = []
    can_deeps = []

    is_inc = False
    for i, yi in enumerate(y):
        if prev is None:
            prev = yi
        else:
            grad = yi - prev
            if grad > 0:
                if not is_inc:
                    cans.append(i - 1)
                    can_deeps.append(prev)
                    is_inc = True
            else:
                is_inc = False
            prev = yi
    return x[np.array(cans)[np.argsort(can_deeps)[:n]]]


def _rfastcormics_fit(x, y):
    mx_pt = np.argmax(y)  # right peak index
    right = y[mx_pt:]
    left = right[-1:0:-1]
    left = np.minimum(left, np.append(np.zeros(max(0, -(2 * mx_pt - 10000 + 1))),
                                      y[max(0, 2 * mx_pt - 10000 + 1): mx_pt]))[max(0, -(2 * mx_pt - 10000 + 1)):]
    exp_curv = np.concatenate([np.zeros(max(0, 10000 - 2 * (10000 - mx_pt) +1)), left, right])
    res_curv = y - exp_curv
    init_vals = [1, 0, 1]
    best_vals_right, covar_right = curve_fit(gaussian, xdata=x, ydata=exp_curv, p0=init_vals)
    best_vals_left, covar_left = curve_fit(gaussian, xdata=x, ydata=res_curv, p0=init_vals)  # loss = least square
    return best_vals_right, best_vals_left, covar_right, covar_left


def _bimodal_fit(x, y):
    c1, c2 = find_canyons(x, y)
    c1, c2 = min(c1, c2), max(c1, c2)
    print("original guess: ", c1, c2)
    init_vals = (50, c1, 1, 150, c2, 1)
    try:
        p, covar = curve_fit(bimodal, xdata=x, ydata=y,
                             p0=init_vals, bounds=((0, -np.inf, 0, 0, -np.inf, 0), np.inf))
    except RuntimeError:
        warnings.warn("Fail to optimize")
        p = init_vals
        covar = None

    A1, mu1, wid1, A2, mu2, wid2 = p
    print("fitted Amps: ", A1, A2)
    print("fitted means: ", mu1, mu2)
    if mu1 < mu2:
        return (A2, mu2, wid2,), (A1, mu1, wid1,), covar
    return (A1, mu1, wid1,), (A2, mu2, wid2,), covar


def get_PROM_threshold(df, q=0.33) -> float:
    return np.quantile(df.values, q)


def get_rfastcormics_thresholds(dat: np.ndarray,
                                cut_off: float = -np.inf,
                                file_name: str = None,
                                plot_dist: bool = False):
    dat = dat[dat > cut_off]
    kde_f = gaussian_kde(dat)
    x = np.linspace(dat.min(), dat.max(), 10000)
    y = kde_f(x)
    best_vals_right, best_vals_left, _ = _bimodal_fit(x, y)
    sigma = np.sqrt(abs(best_vals_right[2]) / 2)
    zscored_x = (x - best_vals_right[1]) / sigma
    zscored_left_thres = (best_vals_left[1] - best_vals_right[1]) / sigma  # if the threshold is lower than -3, pick -3
    if plot_dist:
        right_c = gaussian(x, *tuple(best_vals_right))
        left_c = gaussian(x, *tuple(best_vals_left))
        fig, axes = plt.subplots(1, 2, figsize=(11, 6))
        axes[0].plot(zscored_x, y, label="Data")
        axes[0].plot(zscored_x, right_c, label="Fitted expressed distribution")
        axes[0].plot(zscored_x, left_c, label="Fitted non-expressed distribution")
        axes[0].plot([0, 0], [0, np.max(y)])
        axes[0].plot([zscored_left_thres, zscored_left_thres], [0, np.max(y)])
        axes[0].legend()

        axes[1].plot(x, y, label="Data")
        axes[1].plot(x, right_c, label="Fitted expressed distribution")
        axes[1].plot(x, left_c, label="Fitted non-expressed distribution")
        axes[1].plot([best_vals_left[1], best_vals_left[1]], [0, np.max(y)])
        axes[1].plot([best_vals_right[1], best_vals_right[1]], [0, np.max(y)])
        axes[1].legend()
        if file_name:
            plt.savefig(file_name, dpi=300)
        plt.show()
    print("Fitted values: ")
    print(best_vals_right[1], best_vals_left[1])
    return best_vals_right[1], best_vals_left[1]