import numpy as np

from pipeGEM.core import Problem


def test_problem_extension(ecoli):
    p = Problem(model=ecoli)
    m, n = len(ecoli.metabolites), len(ecoli.reactions)
    e_S = np.zeros(shape=(m, n))
    np.fill_diagonal(e_S, 1)
    e_v = np.array(["C" for _ in range(n)])
    e_lbs = np.zeros((n,))
    e_ubs = np.ones((n,))
    p.extend_horizontal(e_S, e_v, e_lbs, e_ubs)
    mod = p.to_model("new")
    print(mod.optimize())
    print(mod.get_problem_fluxes())

    # e_S = np.zeros(shape=(m, 2 * n))
    # np.fill_diagonal(e_S, 1)
    # e_b = np.zeros((m * 2,))
    # p.extend_vertical(e_S, e_b)
    #
    # print(mod.optimize())
    # print(mod.get_problem_fluxes())
