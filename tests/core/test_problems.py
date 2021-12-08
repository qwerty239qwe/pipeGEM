import numpy as np

from pipeGEM.core import Problem
from pipeGEM.analysis import ProblemAnalyzer
from pipeGEM.integration.algo import BlockedProblem


class TestProblem(Problem):
    def __init__(self, model):
        super().__init__(model)

    def modify_problem(self) -> None:
        m, n = len(self.model.metabolites), len(self.model.reactions)
        e_S = np.zeros(shape=(m, n))
        #np.fill_diagonal(e_S, 1)
        e_v = np.array(["C" for _ in range(n)])
        e_lbs = -np.ones((n,)) * np.inf
        e_ubs = np.ones((n,)) * np.inf
        self.extend_horizontal(e_S, e_v, e_lbs, e_ubs, e_objs=np.ones(n,))
        e_S = np.zeros(shape=(m, 2 * n))
        #np.fill_diagonal(e_S, 1)
        e_b = np.zeros((m,))
        self.extend_vertical(e_S, e_b)


class TestProblem2(Problem):
    def __init__(self, model):
        super().__init__(model)

    def modify_problem(self) -> None:
        pass


def test_problem_consistent(ecoli_core):
    p = TestProblem2(model=ecoli_core)
    mod = ProblemAnalyzer(p)
    sol = ecoli_core.optimize().to_frame()
    new_sol = mod.get_fluxes()
    assert all(np.equal(sol["fluxes"], new_sol["fluxes"]))


def test_problem_extension(ecoli_core):
    p = TestProblem(model=ecoli_core)
    p2 = TestProblem2(model=ecoli_core)
    n = len([r for r in ecoli_core.reactions])

    mod = ProblemAnalyzer(p)
    mod2 = ProblemAnalyzer(p2)
    np.equal(mod.get_fluxes()[:n], mod2.get_fluxes()[:n])


def test_BlockedProblem(ecoli_core):
    bp = BlockedProblem(ecoli_core)
    b = ProblemAnalyzer(bp)
    sol = b.get_fluxes("min")
    print(sol[sol["fluxes"] != 0])
    print(sol[sol == -1].index)