import numpy as np


def sigmoid(x, coef):
    return 1 / (1 + np.exp(-coef * x))


def log_xplus1(x, coef=1):
    return np.log10(x + coef)


def exp_x(x, coef=1, a=1.5):
    return a ** x - coef

