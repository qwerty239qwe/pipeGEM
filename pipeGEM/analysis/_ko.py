import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time

from typing import Optional
from cobra.flux_analysis.deletion import single_gene_deletion
from cobra.flux_analysis.parsimonious import pfba
from cobra.flux_analysis.moma import moma
from cobra.core.solution import Solution
from pipeGEM.utils import ObjectFactory
from pipeGEM.analysis.results import Single_KO_Analysis, KO_Analysis


def ko_once(model, g, method="FBA"):
    with model:
        model.genes.get_by_id(g).knock_out()
        if method == "FBA":
            sol = model.optimize()
        elif method == "pFBA":
            sol = pfba(model)
        elif method == "moma":
            sol = moma(model)
        else:
            raise ValueError(method, " is not supported")
    return sol


def get_ko_df(model, gene_list, method="FBA"):
    results = {}
    for g in tqdm(gene_list):
        sol = ko_once(model, g, method)
        results[g] = sol.to_frame()["fluxes"]
        results[g]["status"] = sol.status
        results[g]["objective_value"] = sol.objective_value
    return pd.DataFrame(results)


class KO_Analyzers(ObjectFactory):
    def __init__(self):
        super().__init__()


class KO_Impact:
    def __init__(self,
                 gene_ids,
                 affected_rxns: dict):
        self._affected_rxns = pd.Series(affected_rxns).dropna()
        self._affected_rxns.sort_index()
        self.gene_ids = gene_ids

    @property
    def affected_rxns(self):
        return self._affected_rxns

    def __eq__(self, other):
        return all(self.affected_rxns == other.affected_rxns)

    def __hash__(self):
        return hash(self.affected_rxns.values.tobytes())


class KO_Analyzer:
    def __init__(self,
                 model,
                 analysis_obj: "KO_Analysis",
                 solver = "glpk"):
        self.model = model
        self.model.cobra_model.solver = solver
        self._solver_name = solver
        self._analysis_obj = analysis_obj
        self._ko_impacts = {}

    @property
    def solver_name(self):
        return self._solver_name

    def get_ko_impact(self, gene_ids, **kwargs) -> KO_Impact:
        return KO_Impact(gene_ids=gene_ids,
                         affected_rxns=self.model.simulate_ko_genes(gene_ids=gene_ids, **kwargs))

    def analysis_func(self, **kwargs):
        raise NotImplementedError

    def analyze(self, **kwargs) -> "KO_Analysis":
        self._analysis_obj.add_result(self.analysis_func(**kwargs))
        return self._analysis_obj


class Single_KO_Analyzer(KO_Analyzer):
    def __init__(self, model, solver, log=None):
        super().__init__(model=model,
                         solver=solver,
                         analysis_obj=Single_KO_Analysis(log=log))

    def analysis_func(self,
                      method: str = "FBA",
                      solution: Optional["Solution"] = None,
                      processes: Optional[int] = 1,
                      **kwargs):
        s = time()
        result_df = get_ko_df(model=self.model.cobra_model,
                              gene_list=[g.id for g in self.model.genes],
                              method=method)
        print(f"KO finished in {time() - s} secs.")
        return result_df


ko_analyzers = KO_Analyzers()
ko_analyzers.register("single_KO", Single_KO_Analyzer)