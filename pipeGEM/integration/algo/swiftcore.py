import numpy as np
import numpy.ma as ma
import cobra
from scipy.linalg import qr, norm

from ._LP import BlockedProblem, Problem


def swiftcc(model,
            tol = 2.2204e-16,
            return_model=False):
    blk_p = BlockedProblem(model=model)
    rev = blk_p.get_rev().copy()
    S: np.ndarray = blk_p.S.copy()
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
    def __init__(self, model, consistent, weights, reduction):
        super().__init__(model)
        self.consistent, self.weights = consistent, weights
        self.reduction = reduction

    def _reduction(self, rev):
        S = self.S
        m, n = S.shape
        reduction = True
        while reduction:
            reduction = False
            for i in range(m - 1, -1, -1):
                if i < S.shape[1]:
                    non_zero_cols = np.argwhere(S[i, :] > 0).flatten()
                    if len(non_zero_cols) == 2:
                        c = S[i, non_zero_cols[0]] / S[i, non_zero_cols[1]]
                        if c < 0:
                            if rev[non_zero_cols[1]] != 1:
                                rev[non_zero_cols[0]] = rev[non_zero_cols[1]]
                        else:
                            if rev[non_zero_cols[1]] != 1:
                                rev[non_zero_cols[0]] = -1 - rev[non_zero_cols[1]]
                        self.lbs[non_zero_cols[0]] = max([self.lbs[non_zero_cols[0]], -self.lbs[non_zero_cols[1]] / c])
                        self.ubs[non_zero_cols[0]] = max([self.ubs[non_zero_cols[0]], -self.ubs[non_zero_cols[1]] / c])
                        rev = np.delete(rev, non_zero_cols[1])
                        self.lbs = np.delete(self.lbs, non_zero_cols[1])
                        self.ubs = np.delete(self.ubs, non_zero_cols[1])
                        self.S[:, non_zero_cols[0]] = self.S[:, non_zero_cols[0]] - c * self.S[:, non_zero_cols[1]]
                        self.S = np.delete(self.S, non_zero_cols[1], 1)
                        # TODO: delete S rows with zeros only, mod coupling, weights, ...etc, maybe use maskarray


    def _flip(self):
        rev = np.ones(self.S.shape[1])
        rev[self.lbs == 0], rev[self.ubs == 0] = 0, -1
        self.ubs[rev == -1] = -self.lbs[rev == -1]
        self.lbs[rev == -1] = 0
        self.ubs /= np.norm(self.ubs)
        self.lbs /= np.norm(self.lbs)
        self.S[:, rev == -1] = -self.S[:, rev == -1]
        self.temp_rev = rev

    def modify_problem(self) -> None:
        self._flip()
        m, n = self.S.shape
        rev = self.get_rev()
        dense = np.zeros(shape=(n,))
        dense[~self.consistent] = np.random.rand(n)
        k_v, l_v = (self.weights != 0) & rev, (self.weights != 0) & ~rev
        k, l = np.sum(k_v), np.sum(l_v)
        self.objs = dense
        self.lbs = np.where(self.consistent, -np.inf, self.lbs)
        self.lbs[l_v] = 0
        if all(self.consistent):
            self.lbs[(self.weights == 0) & (~rev)] = 1
        self.ubs = np.where(self.consistent, np.inf, self.ubs)
        self.extend_horizontal(np.zeros(shape=(m, k + l)), e_v=np.array(["C" for _ in range(k + l)]),
                               e_v_lb=-np.ones(shape=(k + l)) * np.inf, e_v_ub=np.ones(shape=(k + l)) * np.inf,
                               e_objs=np.concatenate([self.weights[k_v], self.weights[l_v]]),
                               e_names=[f"ext_const_{i}" for i in range(m + l)])
        temp1, temp2 = np.eyes(n), np.eyes(k + l)
        btm_ext_S_1 = np.concatenate([temp1[k_v, :], temp2[k_v, :]], axis=1)
        btm_ext_S_2 = np.concatenate([-temp1[self.weights != 0, :], temp2], axis=1)
        csense = np.array(["G" for _ in range(2 * k + l)])
        b = np.zeros(2 * k + l)
        self.extend_vertical(e_S=np.concatenate([btm_ext_S_1, btm_ext_S_2], axis=0), e_b=b, e_c=csense,
                             e_names=[f"ext_var_{i}" for i in range(2 * k + l)])


def _core(prob: Problem, consistent, weights):
    # TODO: delete this
    prob = prob.copy()
    m, n = prob.S.shape
    rev = prob.get_rev()
    dense = np.zeros(shape=(n,))
    dense[~consistent] = np.random.rand(n)
    k_v, l_v = (weights != 0) & rev, (weights != 0) & ~rev
    k, l = np.sum(k_v), np.sum(l_v)
    prob.objs = dense
    prob.lbs = np.where(consistent, -np.inf, prob.lbs)
    prob.lbs[l_v] = 0
    if all(consistent):
        prob.lbs[(weights == 0) & (~rev)] = 1
    prob.ubs = np.where(consistent, np.inf, prob.ubs)
    prob.extend_horizontal(np.zeros(shape=(m, k+l)), e_v=np.array(["C" for _ in range(k + l)]),
                           e_v_lb=-np.ones(shape=(k+l)) * np.inf, e_v_ub=np.ones(shape=(k+l)) * np.inf,
                           e_objs=np.concatenate([weights[k_v], weights[l_v]]),
                           e_names=[f"ext_const_{i}" for i in range(m + l)])
    temp1, temp2 = np.eyes(n), np.eyes(k+l)
    btm_ext_S_1 = np.concatenate([temp1[k_v, :], temp2[k_v, :]], axis=1)
    btm_ext_S_2 = np.concatenate([-temp1[weights != 0, :], temp2], axis=1)
    csense = np.array(["G" for _ in range(2 * k + l)])
    b = np.zeros(2 * k + l)
    prob.extend_vertical(e_S=np.concatenate([btm_ext_S_1, btm_ext_S_2], axis=0), e_b=b, e_c=csense,
                         e_names=[f"ext_var_{i}" for i in range(2 * k + l)])
    return prob


def swiftcore(model, core_index, weights=None, reduction=False):
    if weights is None:
        weights = np.ones(shape=(len(model.reactions),))
    consistent = swiftcc(model)
    react_num = np.arange(len(model.reactions))
    couplings = np.arange(len(model.reactions))
    problem = CoreProblem(model=model, consistent=consistent, weights=weights, reduction=reduction)
    flux = problem.to_model("core").get_problem_fluxes()
    # TODO: finish this
