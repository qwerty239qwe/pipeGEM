import numpy as np


def sigmoid(x, coef):
    return 1 / (1 + np.exp(-coef * x))


def mod_log10(x, coef=1001) -> float:
    return np.log10(x + coef)


def exp_x(x, coef=1, a=1.1):
    return a ** x - coef

