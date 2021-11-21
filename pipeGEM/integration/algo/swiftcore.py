import numpy as np
import numpy.ma as ma
import cobra
from scipy.linalg import qr, norm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from ._LP import BlockedProblem, Problem
from pipeGEM.utils import get_rev_arr


def swiftcc(model,
            tol = 2.2204e-16,
            return_model=False):
    blk_p = BlockedProblem(model=model)
    rev = get_rev_arr(model)
    S = cobra.util.create_stoichiometric_matrix(model)
    blk = blk_p.to_model("block")
    blk.optimize()
    consistent = np.array([True for _ in range(model.n_rxns)])
    sol = blk.get_problem_fluxes()
    sol = sol.iloc[model.n_mets:, :]
    consistent[sol["fluxes"] < -0.5] = False
    tol = tol * norm(S[:, consistent])
    q, r = qr(a=S[:, consistent].T)
    z = q[rev[consistent], np.sum(np.abs(np.diag(r)) > tol):]
    consistent[rev & consistent] = np.diag(z @ z.T) > (tol ** 2)
    if return_model:
        output_model = model.copy()
        rxns = np.array([r.id for r in model.reactions])
        output_model.remove_reactions(rxns[~consistent], remove_orphans=True)
        return output_model
    return consistent


class CoreProblem(Problem):
    def __init__(self, model, consistent, core_index, weights, do_flip, do_reduction, **kwargs):
        self.consistent, self.weights = consistent, weights
        self.core_index = core_index
        self.do_reduction = do_reduction
        self.do_flip = do_flip
        self.original_S_shape = None
        super().__init__(model, **kwargs)

    def _reduction(self, rev, couplings, react_num, core_index, weights):
        S = self.S
        m, n = S.shape
        reduction = True
        col_mask = np.array([True for _ in range(S.shape[1])])
        row_mask = np.array([True for _ in range(S.shape[0])])
        while reduction:
            reduction = False
            for i in range(m):
                if not row_mask[i]:
                    continue
                non_zero_cols = np.argwhere(S[i, :][:, col_mask] > 0).flatten()
                if len(non_zero_cols) == 2:
                    c = S[i, non_zero_cols[0]] / S[i, non_zero_cols[1]]
                    if c < 0:
                        if rev[non_zero_cols[1]] != 1:
                            rev[non_zero_cols[0]] = rev[non_zero_cols[1]]
                    else:
                        if rev[non_zero_cols[1]] != 1:
                            rev[non_zero_cols[0]] = -1 - rev[non_zero_cols[1]]
                    self.lbs[non_zero_cols[0]] = max([self.lbs[non_zero_cols[0]], -self.lbs[non_zero_cols[1]] / c])
                    self.ubs[non_zero_cols[0]] = min([self.ubs[non_zero_cols[0]], -self.ubs[non_zero_cols[1]] / c])
                    col_mask[non_zero_cols[1]] = False
                    S[row_mask, non_zero_cols[0]] = S[row_mask, non_zero_cols[0]] - c * S[row_mask, non_zero_cols[1]]
                    row_mask &= (S[:, col_mask].sum(axis=1) > 0)
                    couplings[couplings == react_num[non_zero_cols[1]]] = react_num[non_zero_cols[0]]
                    core_index[react_num[non_zero_cols[0]]] |= core_index[react_num[non_zero_cols[1]]]
                    # core_index[core_index == react_num[non_zero_cols[1]]] = react_num[non_zero_cols[0]]
                    weights[non_zero_cols[0]] += weights[non_zero_cols[1]]
                    reduction = True
        self.couplings = couplings
        self._couple_rxns(col_mask, row_mask, react_num, core_index, weights)

    def _couple_rxns(self, col_mask, row_mask, react_num, core_index, weights):
        self.S = self.S[row_mask, :][:, col_mask]
        self.weights = weights[col_mask]
        self.lbs, self.ubs = self.lbs[col_mask], self.ubs[col_mask]
        self.react_num = react_num[col_mask]
        self.core_index = core_index[col_mask]

    def _flip(self):
        rev = np.ones(self.S.shape[1])
        rev[self.lbs == 0], rev[self.ubs == 0] = 0, -1
        self.ubs[rev == -1] = -self.lbs[rev == -1]
        self.lbs[rev == -1] = 0
        self.ubs /= norm(self.ubs)
        self.lbs /= norm(self.lbs)
        self.react_num = np.arange(self.S.shape[1])
        self.couplings = np.arange(self.S.shape[1])
        if self.do_reduction:
            self._reduction(rev,
                            self.couplings,
                            self.react_num,
                            core_index=self.core_index,
                            weights=self.weights)
        self.S[:, rev == -1] = -self.S[:, rev == -1]
        self.temp_rev = rev

    def modify_problem(self) -> None:
        m, n = self.S.shape
        print(m, n)
        self.original_S_shape = (m, n)
        if self.do_flip:
            self._flip()

        rev = self.get_rev()
        dense = np.zeros(shape=(n,))
        dense[~self.consistent] = np.random.rand(sum(~self.consistent))
        k_v, l_v = (self.weights != 0) & rev, (self.weights != 0) & (~rev)
        print(len(k_v), len(l_v), len(rev), len(self.weights))
        k, l = np.sum(k_v), np.sum(l_v)
        self.objs = dense
        self.lbs = np.where(self.consistent, -np.inf, self.lbs)
        self.lbs[l_v] = 0
        self.v = np.array(["C" for _ in range(n)])
        self.c = np.array(["E" for _ in range(m)])
        self.b = np.zeros(m)
        if all(self.consistent):
            self.lbs[(self.weights == 0) & (~rev)] = 1
        self.ubs = np.where(self.consistent, np.inf, self.ubs)
        self.extend_horizontal(np.zeros(shape=(m, k + l)),
                               e_v=np.array(["C" for _ in range(k + l)]),
                               e_v_lb=-np.ones(shape=(k + l)) * np.inf,
                               e_v_ub=np.ones(shape=(k + l)) * np.inf,
                               e_objs=np.concatenate([self.weights[k_v], self.weights[l_v]]),
                               e_names=[f"ext_const_{i}" for i in range(k + l)])
        temp1, temp2 = np.eye(n), np.eye(k + l)
        btm_ext_S_1 = np.concatenate([temp1[k_v, :], temp2[rev[self.weights!=0] == 1, :]], axis=1)
        btm_ext_S_2 = np.concatenate([-temp1[self.weights != 0, :], temp2], axis=1)
        csense = np.array(["G" for _ in range(2 * k + l)])
        b = np.zeros(2 * k + l)
        self.extend_vertical(e_S=np.concatenate([btm_ext_S_1, btm_ext_S_2], axis=0),
                             e_b=b,
                             e_c=csense,
                             e_names=[f"ext_var_{i}" for i in range(2 * k + l)])


def swiftCore(model, core_index, weights=None, reduction=False, k=10, tol=1e-10):
    if weights is None:
        weights = np.ones(shape=(len(model.reactions),))
    consistent = swiftcc(model)
    is_core = np.array([(i in core_index) for i in range(len(model.reactions))])
    __w = len(is_core)
    print(len(is_core))
    m, n = len(model.metabolites), len(model.reactions)
    problem = CoreProblem(model=model,
                          consistent=consistent,
                          weights=weights,
                          core_index=is_core,
                          do_flip=True,
                          do_reduction=reduction)

    rxn_num, coupling = problem.react_num, problem.couplings
    core_model = problem.to_model("core")
    flux = core_model.get_problem_fluxes()
    weights = problem.weights
    assert len(weights) == __w
    weights[problem.core_index] = 0
    assert len(weights) == __w
    m_, n_ = problem.original_S_shape
    flux = flux[:n_]
    weights[abs(flux["fluxes"].values) > tol] = 0
    assert len(weights) == __w
    blocked = np.array([False for _ in range(n_)])
    if n == n_:
        blocked = problem.core_index.copy()
        blocked[abs(flux["fluxes"].values) > tol] = False
    else:
        _, D, Vt = svds(csr_matrix(problem.S[:m_, :n_][:, weights == 0]), k=k, which="SM")
        Vt = Vt[np.diag(D) < tol * norm(problem.S[:m_, :n_][:, weights == 0]), :]
        blocked[weights == 0] = np.all(abs(Vt) < tol, 0)

    while np.any(blocked):
        blocked_size = sum(blocked)
        problem = CoreProblem.from_problem(problem,
                                           s_shape=("m", "n"),
                                           consistent=consistent,
                                           weights=weights,
                                           core_index=is_core,
                                           do_flip=False,
                                           do_reduction=False)
        core_model = problem.to_model("core")
        flux = core_model.get_problem_fluxes()[:n_]
        weights[abs(flux["fluxes"].values) > tol] = 0
        assert len(weights) == __w
        blocked[abs(flux["fluxes"].values) > tol] = 0
        if 2 * sum(blocked) > blocked_size:
            weights /= 2
    kept_rxns = rxn_num[(weights == 0)]
    rxns_to_remove = [r.id for r in model.reactions if r in kept_rxns]
    return model.copy().remove_reactions(rxns_to_remove, remove_orphans=True)