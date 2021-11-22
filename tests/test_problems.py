import numpy as np

from pipeGEM.core import Problem
from pipeGEM.integration.algo import BlockedProblem


class TestProblem(Problem):
    def __init__(self, model):
        super().__init__(model)

    def modify_problem(self) -> None:
        m, n = len(self.model.metabolites), len(self.model.reactions)
        e_S = np.zeros(shape=(m, n))
        np.fill_diagonal(e_S, 1)
        e_v = np.array(["C" for _ in range(n)])
        e_lbs = np.zeros((n,))
        e_ubs = np.ones((n,))
        self.extend_horizontal(e_S, e_v, e_lbs, e_ubs)

        e_S = np.zeros(shape=(m, 2 * n))
        np.fill_diagonal(e_S, 1)
        e_b = np.zeros((m,))
        self.extend_vertical(e_S, e_b)


class TestProblem2(Problem):
    def __init__(self, model):
        super().__init__(model)

    def modify_problem(self) -> None:
        pass


def test_problem_consistent(ecoli_core):
    p = TestProblem2(model=ecoli_core)
    mod = p.to_model("new")
    sol = ecoli_core.optimize().to_frame()
    new_sol = mod.get_problem_fluxes()
    assert np.equal(sol["fluxes"], new_sol["fluxes"])


def test_problem_extension(ecoli):
    p = TestProblem(model=ecoli)
    mod = p.to_model("new")
    print(mod.optimize())
    print(mod.get_problem_fluxes())


def test_BlockedProblem(ecoli):
    bp = BlockedProblem(ecoli)
    b = bp.to_model("bp")
    b.optimize()
    sol = b.get_problem_fluxes()
    print(sol[sol == -1].index)