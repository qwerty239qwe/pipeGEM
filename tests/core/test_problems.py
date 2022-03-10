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
        e_v = np.array(["C" for _ in range(n)])
        e_lbs = -np.ones((n,)) * 1e8
        e_ubs = np.ones((n,)) * 1e8
        self.extend_horizontal(e_S, e_v, e_lbs, e_ubs, e_objs=np.ones(n,))
        e_S = np.zeros(shape=(m, 2 * n))
        e_b = np.zeros((m,))
        self.extend_vertical(e_S, e_b)


class TestProblem2(Problem):
    def __init__(self, model):
        super().__init__(model)

    def modify_problem(self) -> None:
        pass


def test_problem_flux_consistency(ecoli_core):
    solver = "glpk"
    p = TestProblem2(model=ecoli_core)
    mod = ProblemAnalyzer(p, solver=solver)
    ecoli_core.solver = solver
    sol = ecoli_core.optimize().to_frame()
    new_sol = mod.get_fluxes()
    diff = sol["fluxes"] - new_sol["fluxes"]

    assert all(diff < 1e-8), diff[diff > 1e-8]


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