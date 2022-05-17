import warnings
from pathlib import Path

from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import cobra

from pipeGEM.integration.mapping import Expression


__all__ = ["get_rfastcormics_thresholds", "get_PROM_threshold",
           "find_exp_threshold", "get_expression_thresholds", "get_discretize_data"]


def gaussian(x, amp, cen, wid):
    return amp * np.exp(-np.power(x-cen, 2) / wid)


def bimodal(x, A1, mu1, wid1, A2, mu2, wid2):
    return gaussian(x, A1, mu1, wid1)+gaussian(x, A2, mu2, wid2)


def get_second_deriv(y, dx):
    return np.gradient(np.gradient(y, dx), dx)


def find_canyons(x, y, min_x_dis=3):
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

    candidates = x[np.array(cans)[np.argsort(can_deeps)]]
    # greedy
    first_sel = candidates[0]
    cur_best_dis, cur_best_c = 0, None
    for c in candidates[1:]:
        if first_sel - c >= min_x_dis:
            return first_sel, c
        if cur_best_dis < first_sel - c:
            cur_best_dis = first_sel - c
            cur_best_c = c

    return first_sel, cur_best_c


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


def _check_fitted_param(p, amp_ratio_tol=4, var_ratio_tol=2, mean_diff_tol=4):
    amp1, mean1, cov1, amp2, mean2, cov2 = p
    if abs(np.log2(amp1) - np.log2(amp2)) > np.log2(amp_ratio_tol):
        return False
    if abs(np.log2(cov1) - np.log2(cov1)) > np.log2(var_ratio_tol):
        return False
    if abs(mean1 - mean2) > mean_diff_tol:
        return False
    return True


def _get_y_by_nearest_x(x, y, c):
    dtx = abs(x - c)
    return y[np.argsort(dtx)[0]]


def _bimodal_fit(x, y, amp_ratio_tol=4, var_ratio_tol=2, mean_diff_tol=4):
    c1, c2 = find_canyons(x, y, min_x_dis=mean_diff_tol)
    c1, c2 = min(c1, c2), max(c1, c2)
    print("original guess: ", c1, c2)
    init_vals = (1, c1, 1, 1, c2, 1)
    init_val_displayed = (_get_y_by_nearest_x(x, y, c1), c1, 1, _get_y_by_nearest_x(x, y, c2), c2, 1)
    grid = [(5 + 20 * i, 5 + 20 * j) for i in range(0, 4) for j in range(0, 4)]
    try:
        found_best = False
        it = 0
        while (not found_best) and it < len(grid):
            tried_vals = [grid[it][0], init_vals[1], init_vals[2],
                          grid[it][1], init_vals[4], init_vals[5]]
            p, covar = curve_fit(bimodal, xdata=x, ydata=y,
                                 p0=tried_vals, bounds=((0, -np.inf, 0, 0, -np.inf, 0), np.inf))
            if _check_fitted_param(p,
                                   amp_ratio_tol=amp_ratio_tol,
                                   var_ratio_tol=var_ratio_tol,
                                   mean_diff_tol=mean_diff_tol):
                found_best = True
            else:
                it += 1
        if not found_best:
            warnings.warn("Fail to find proper parameters, use initial guess")
            p = init_val_displayed
            covar = None

    except RuntimeError:
        warnings.warn("Fail to optimize")
        p = init_val_displayed
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
                                plot_dist: bool = False,
                                use_first_guess: bool = False):
    dat = dat[dat > cut_off]
    kde_f = gaussian_kde(dat)
    x = np.linspace(dat.min(), dat.max(), 10000)
    y = kde_f(x)
    if use_first_guess:
        c1, c2 = find_canyons(x, y)
        c1, c2 = min(c1, c2), max(c1, c2)
        best_vals_right, best_vals_left = (1, c2, 1), (1, c1, 1)
    else:
        best_vals_right, best_vals_left, _ = _bimodal_fit(x, y)
    sigma = np.sqrt(abs(best_vals_right[2]) / 2)
    zscored_x = (x - best_vals_right[1]) / sigma
    zscored_left_thres = (best_vals_left[1] - best_vals_right[1]) / sigma  # if the threshold is lower than -3, pick -3
    if plot_dist:

        fig, ax = plt.subplots(figsize=(8, 6))
        # axes[0].plot(zscored_x, y, label="Data")
        # axes[0].plot(zscored_x, right_c, label="Fitted expressed distribution")
        # axes[0].plot(zscored_x, left_c, label="Fitted non-expressed distribution")
        # axes[0].plot([0, 0], [0, np.max(y)])
        # axes[0].plot([zscored_left_thres, zscored_left_thres], [0, np.max(y)])
        # axes[0].legend()

        ax.plot(x, y, label="Data")
        if not use_first_guess:
            right_c = gaussian(x, *tuple(best_vals_right))
            left_c = gaussian(x, *tuple(best_vals_left))
            ax.plot(x, right_c, label="Fitted expressed distribution")
            ax.plot(x, left_c, label="Fitted non-expressed distribution")
        ax.plot([best_vals_left[1], best_vals_left[1]], [0, np.max(y)])
        ax.plot([best_vals_right[1], best_vals_right[1]], [0, np.max(y)])
        ax.legend()
        if file_name:
            plt.savefig(file_name, dpi=300)
        plt.show()
    print("Fitted values: ")
    print(best_vals_right[1], best_vals_left[1])
    return best_vals_right[1], best_vals_left[1]


def find_exp_threshold(model: cobra.Model,
                       gene_data_dict: dict,
                       transform=np.log2,
                       threshold=1e-4,
                       plot_dist=False):
    if transform:
        threshold = transform(threshold)
        trans_data_dict = {i: transform(v) if v != 0 else threshold - 1 for i, v in gene_data_dict.items()}
        trans_data_arr = np.array(list(trans_data_dict.values()))
    else:
        trans_data_dict = gene_data_dict.copy()
        trans_data_arr = np.array(list(trans_data_dict.values()))
    expr = Expression(model, trans_data_dict, expression_threshold=threshold, method='mCADRE')
    rxn_scores_dict = expr.rxn_scores  # want to recognize no gene related rxns and zero expr rxns
    hi, lo = get_rfastcormics_thresholds(trans_data_arr, threshold, plot_dist=plot_dist)
    for rxn, exp in rxn_scores_dict.items():
        if exp == 0:
            rxn_scores_dict[rxn] = hi

    return hi, lo, rxn_scores_dict


def get_expression_thresholds(data_df,
                              sample_names,
                              cut_off=-np.inf,
                              naming_format="./thresholds/{sample_name}.png",
                              plot_dist=True):
    expr, nexpr = {}, {}
    if naming_format is not None:
        Path(naming_format).parent.mkdir(parents=True, exist_ok=True)

    if sample_names == "all":
        sample_names = data_df.columns.to_list()

    for sample_name in sample_names:
        expr[sample_name], nexpr[sample_name] = get_rfastcormics_thresholds(data_df[sample_name].values,
                                                                            cut_off=cut_off,
                                                                            file_name=naming_format.format(sample_name=sample_name)
                                                                            if naming_format is not None else None,
                                                                            plot_dist=plot_dist)
    return {"expr_threshold_dic": expr,
            "non_expr_threshold_dic": nexpr,
            "data_df": data_df}


def get_discretize_data(sample_names,
                        data_df,
                        expr_threshold_dic,
                        non_expr_threshold_dic):
    disc_data = data_df.copy()
    for sample_name in sample_names:
        exp_thres, nexp_thres = expr_threshold_dic[sample_name], non_expr_threshold_dic[sample_name]
        disc_data[sample_name] = disc_data[sample_name].apply(lambda x: 1
                                                              if x >= exp_thres else -1 if x <= nexp_thres else 0)

    return disc_data