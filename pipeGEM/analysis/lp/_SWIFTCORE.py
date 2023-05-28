import numpy as np
from pipeGEM.analysis import Problem


class BlockedProblem(Problem):
    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)

    def _transpose_S(self):
        m, n = self.S.shape
        self._irrev = ~self.get_rev()
        self.S = self.S.T
        self.objs = np.zeros((m,))
        max_val = 1e3 * max(m, n) * 3
        self.lbs = np.array([-max_val for _ in range(m)])
        self.ubs = np.array([max_val for _ in range(m)])
        self.b = np.zeros((n,))
        self.c = np.array(["E" for _ in range(n)])
        self.c[self._irrev] = "L"
        self.v = np.array(["C" for _ in range(m)])
        self.col_names, self.row_names = self.row_names, self.col_names
        self._check_matrix()

    def modify_problem(self):
        m, n = self.S.shape
        self._transpose_S()

        # extend right
        ext_objs = np.zeros((n,))
        ext_objs[self._irrev] = 1
        ext_S = -np.eye(n)
        e_lbs = np.zeros((n,))
        e_ubs = np.zeros((n,))
        e_lbs[self._irrev] = -1
        e_v = np.array(["C" for _ in range(n)])
        names = [f"rxn_{i}" for i in range(n)]
        self.extend_horizontal(ext_S, e_v, e_lbs, e_ubs, ext_objs, names)