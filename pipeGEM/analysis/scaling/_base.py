from pipeGEM.utils import ObjectFactory
from pipeGEM.analysis import ModelScalingResult
from cobra.util.array import create_stoichiometric_matrix
import numpy as np
import numpy.ma as ma


class ModelScalerCollection(ObjectFactory):
    def __init__(self):
        super().__init__()


class ModelScaler:
    def __init__(self):
        super().__init__()
        self._rescaled_A = None
        self._diff_A = None
        self.m_ind = None
        self.r_ind = None
        self.cons_scale_diags = None
        self.mets_scale_diags = None

    def get_R(self, A: np.ndarray):
        raise NotImplementedError()

    def get_S(self, A: np.ndarray):
        raise NotImplementedError()

    def one_operation(self, arr):
        R = self.get_R(arr)
        arr = R @ arr
        self.mets_scale_diags /= np.diag(R)
        S = self.get_S(arr)
        self.cons_scale_diags /= np.diag(S)
        return arr @ S

    def reach_stop_crit(self,
                        arr,
                        ub = 1,
                        lb = -1):
        return np.all((arr <= ub) & (arr >= lb))

    def calc_coef_scale_diff(self, A, log_scale_diff=6):
        masked = ma.array(abs(A), mask=(A == 0))
        row_prob_scales = (np.log10(masked.max(axis=1) / masked.min(axis=1)) >= log_scale_diff).sum()
        col_prob_scales = (np.log10(masked.max(axis=0) / masked.min(axis=0)) >= log_scale_diff).sum()
        return row_prob_scales, col_prob_scales

    def get_rescale_factor(self, model, n_iter=5):
        arr = create_stoichiometric_matrix(model)
        old_arr = arr.copy()
        row_psc, col_psc = self.calc_coef_scale_diff(arr)
        print(f"Problematic rows (metabolite coefficients): {row_psc}")
        print(f"Problematic cols (metabolite coefficients): {col_psc}")
        self.m_ind = model.metabolites.index
        self.r_ind = model.reactions.index
        self.cons_scale_diags = np.ones(shape=(len(model.reactions,)))
        self.mets_scale_diags = np.ones(shape=(len(model.metabolites, )))

        for _ in range(n_iter):
            arr = self.one_operation(arr)
            if self.reach_stop_crit(arr):
                print("Reached stopping criterion")
                break
        row_psc, col_psc = self.calc_coef_scale_diff(arr)
        print(f"Problematic rows (metabolite coefficients): {row_psc}")
        print(f"Problematic cols (reaction coefficients): {col_psc}")
        self._rescaled_A = arr
        self._diff_A = self._rescaled_A - old_arr

    def rescale_model(self, model, n_iter):
        new_mod = model.copy()
        self.get_rescale_factor(new_mod, n_iter)
        rxn_index = {self.r_ind(r): r.id for r in new_mod.reactions}
        met_index = {self.m_ind(m): m for m in new_mod.metabolites}
        for i, m in met_index.items():
            m.id = f"{m.id}_x{self.mets_scale_diags[i]:.3f}"

        for i, r_id in rxn_index.items():
            involved_met_ids = np.where(self._diff_A[:, i] != 0)[0]  # met index
            new_mod.reactions.get_by_id(r_id).add_metabolites({
                met_index[mi]: self._diff_A[mi, i]
                for mi in involved_met_ids
            })
            new_mod.reactions.get_by_id(r_id).lower_bound *= self.cons_scale_diags[i]
            new_mod.reactions.get_by_id(r_id).upper_bound *= self.cons_scale_diags[i]
        result = ModelScalingResult(log=dict(n_iter=n_iter,
                                             method=self.__class__.__name__))
        result.add_result(dict(rescaled_model=new_mod,
                               diff_A=self._diff_A,
                               rescaled_A=self._rescaled_A))
        return result