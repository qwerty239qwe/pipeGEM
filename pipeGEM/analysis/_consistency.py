import pandas as pd
from tqdm import tqdm
import numpy as np
from optlang import Model, Variable, Constraint, Objective

import pipeGEM
from pipeGEM.analysis import timing, LP3, LP7, non_convex_LP7, non_convex_LP3, \
    FVAConsistencyAnalysis, FastCCAnalysis
from ._flux import FVA_Analyzer
from pipeGEM.utils import ObjectFactory
from pipeGEM.utils import get_rxn_set, flip_direction


class FluxLogger:
    def __init__(self):
        self._dfs = []
        self._iter = 0

    def tic(self):
        self._iter += 1

    def add(self, name, flux_series):
        self._dfs.append(flux_series.to_frame(f"{name}_{self._iter}"))

    def to_frame(self):
        return pd.concat(self._dfs, axis=1)


class ConsistencyTesters(ObjectFactory):
    def __init__(self):
        super().__init__()


class ConsistencyTester:
    def __init__(self, model):
        self.model = model
        self._flux_recorder = FluxLogger()

    def analyze(self, **kwargs):
        raise NotImplementedError("")


class StoppingCriteria:
    def __init__(self):
        pass

    def check(self, removed, kept) -> bool:
        raise NotImplementedError("")


class IsInSetStoppingCriteria(StoppingCriteria):
    def __init__(self, ess_set):
        super().__init__()
        self.ess_set = ess_set

    def check(self, removed, kept):
        return any([c in self.ess_set for c in removed])


class NumInequalityStoppingCriteria(StoppingCriteria):
    def __init__(self, var, cons_dict, cons_lb=0):
        super().__init__()
        self.var = var  # {'a': list(rxn_ids)}
        self._kept_var = {k: set() for k, v in self.var.items()}
        self._removed_var = {k: set() for k, v in self.var.items()}
        self.cons_dict = cons_dict
        self.cons_lb = cons_lb

    def _update_var_num(self, new_vars, lbs, n_ubs):
        return {k: Variable(k,
                            ub=len(v),
                            lb=lbs.get(k, 0))
                for k, v in new_vars.items()}

    def check(self, removed, kept) -> bool:
        for k, v in self.var.items():
            for vi in v:
                if vi in kept:
                    self._kept_var[k].add(vi)
                if vi in removed:
                    self._removed_var[k].add(vi)
        variables = self._update_var_num(new_vars=self.var,
                                         lbs={k: len(v) for k, v in self._removed_var.items()},
                                         n_ubs={k: len(v) for k, v in self._kept_var.items()})
        cons = Constraint(sum([variables[k] * v for k, v in self.cons_dict.items()]), lb=self.cons_lb)
        model = Model(name='Ineq')
        model.add([cons])
        status = model.optimize()
        if status != "optimal":
            print("removed", {k: len(v) for k, v in self._removed_var.items()})
            print("kept", {k: len(v) for k, v in self._kept_var.items()})

        return status != "optimal"


class FASTCC(ConsistencyTester):
    def __init__(self, model):
        super().__init__(model)

    @timing
    def analyze(self,
                tol: float,
                return_model: bool = True,
                is_convex=True,
                rxn_scaling_coefs=None,
                stopping_callback=None,
                **kwargs) -> FastCCAnalysis:
        if not is_convex:
            print("Using non-convex fastcc method")
        print(f"Flux tolerance used: {tol}")
        tol_ = tol
        consistent_model = None
        if rxn_scaling_coefs is not None:
            tol = pd.Series({k: tol * v
                             for k, v in rxn_scaling_coefs.items()}).sort_index()
        if return_model:
            consistent_model = self.model.copy()
        all_rxns = get_rxn_set(self.model, dtype=np.array)
        irr_rxns = get_rxn_set(self.model, "irreversible", dtype=np.array)
        no_expressed = get_rxn_set(self.model, "not_expressed", dtype=np.array)
        backward_rxns = get_rxn_set(self.model, "backward", dtype=np.array)
        J = np.setdiff1d(irr_rxns, no_expressed)
        with self.model as model:
            if len(backward_rxns) > 0:
                print(f"Found and flipped {len(backward_rxns)} reactions")
                flip_direction(model, backward_rxns)
            A = np.array(LP7(J, model, tol, use_abs=True, flux_logger=self._flux_recorder))  # rxns to keeps
            print(f"Inconsistent irreversible rxns: {len(np.setdiff1d(J, A))}")

            if len(np.setdiff1d(J, A)) > 0:
                A_2 = np.array(LP7(np.setdiff1d(J, A), model, tol, use_abs=True, flux_logger=self._flux_recorder))
                print(f"Inconsistent irreversible rxns (2nd run): {len(np.setdiff1d(np.setdiff1d(J, A), A_2))}")
                A = np.union1d(A, A_2)

            if stopping_callback is not None:
                reach_stop_crit = False
                for cb in stopping_callback:
                    reach_stop_crit = reach_stop_crit or cb.check(removed=np.union1d(no_expressed, np.setdiff1d(J, A)),
                                                                  kept=A)
                if reach_stop_crit:
                    print("Stopping criteria is met, stop the process.")
                    return FastCCAnalysis(log={"is_convex": is_convex, "tol": tol_, "stopped": True})

            J = np.setdiff1d(all_rxns, np.union1d(np.union1d(A, no_expressed), J))  # rev rxns to check
            # print("J: ", len(J))
            singleton, flipped = False, False
            with tqdm(total=len(J)) as pbar:
                while len(J) != 0:
                    self._flux_recorder.tic()
                    if singleton:
                        Ji = np.array([J[0]])
                        new_supps = np.array(LP3(Ji, model, tol,
                                                 flux_logger=self._flux_recorder)) if is_convex else np.array(
                            non_convex_LP3(Ji, model, tol, flux_logger=self._flux_recorder))
                    else:
                        Ji = J.copy()
                        new_supps = np.array(LP7(Ji, model, tol,
                                                 flux_logger=self._flux_recorder)) if is_convex else np.array(
                            non_convex_LP7(Ji, model, tol, flux_logger=self._flux_recorder))
                    new_kept = np.setdiff1d(new_supps, A)
                    new_removed = []
                    A = np.union1d(A, new_supps)
                    before_n = len(J)
                    J = np.setdiff1d(J, A)
                    after_n = len(J)
                    pbar.update(before_n - after_n)
                    if before_n != after_n:
                        flipped = False
                    else:  # no change in number of rxn_to_keeps
                        Jirev = np.setdiff1d(Ji, irr_rxns)
                        if flipped or len(Jirev) == 0:
                            flipped = False
                            if singleton:
                                new_removed = Ji
                                J = np.setdiff1d(J, Ji)
                                tqdm.write(f"Removed inconsistent reaction: {Ji}")
                                pbar.update(1)
                            else:
                                singleton = True
                        else:
                            flip_direction(model, Jirev)
                            flipped = True
                    if stopping_callback is not None:
                        reach_stop_crit = False
                        for cb in stopping_callback:
                            reach_stop_crit = reach_stop_crit or cb.check(removed=new_removed,
                                                                          kept=new_kept)
                        if reach_stop_crit:
                            print("Stopping criteria is met, stop the process.")
                            pbar.close()
                            return FastCCAnalysis(log={"is_convex": is_convex, "tol": tol_, "stopped": True})

        rxns_to_remove = np.setdiff1d(all_rxns, A)
        if consistent_model is not None:
            consistent_model.remove_reactions(rxns_to_remove, remove_orphans=True)
            if isinstance(consistent_model, pipeGEM.Model):
                consistent_model.rename(name_tag=f"consistent_{self.model.name_tag}")

        result = FastCCAnalysis(log={"is_convex": is_convex, "tol": tol_})
        result.add_result(dict(consistent_model=consistent_model,
                               removed_rxn_ids=rxns_to_remove,
                               kept_rxn_ids=A,
                               flux_record=self._flux_recorder.to_frame()))

        return result


class FVAConsistencyTester(ConsistencyTester):
    def __init__(self, model):
        super().__init__(model)

    @timing
    def analyze(self,
                tol: float,
                return_model: bool = True,
                **kwargs) -> FVAConsistencyAnalysis:
        analyzer = FVA_Analyzer(model=self.model,
                                solver=self.model.solver)
        fva_result = analyzer.analyze(**kwargs)
        rxns_to_remove = fva_result.flux_df.query(f"minimum >= {-tol} and maximum <= {tol}").index.to_list()
        consistent_model = None
        if return_model:
            consistent_model = self.model.copy()
            consistent_model.remove_reactions(rxns_to_remove, remove_orphans=True)
            if isinstance(consistent_model, pipeGEM.Model):
                consistent_model.rename(name_tag=f"consistent_{self.model.name_tag}")

        result = FVAConsistencyAnalysis(log=dict(tol=tol,
                                                 return_model=return_model,
                                                 **kwargs))

        result.add_result(dict(consistent_model=consistent_model,
                               removed_rxn_ids=np.array(rxns_to_remove),
                               kept_rxn_ids=np.array([r.id for r in consistent_model.reactions])))
        return result


consistency_testers = ConsistencyTesters()
consistency_testers.register("FASTCC", FASTCC)
consistency_testers.register("FVA", FVAConsistencyTester)