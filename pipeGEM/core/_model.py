from pathlib import Path
from typing import Union

import pandas as pd

from pipeGEM.core._base import GEMComposite
from pipeGEM.integration.mapping import Expression
from pipeGEM.analysis import FluxAnalyzer


class Model(GEMComposite):
    _is_leaf = True

    def __init__(self,
                 model,
                 name_tag = None,
                 solver = "glpk",
                 data = None):
        super().__init__(name_tag=name_tag)
        self._model = model
        if data is not None:
            self.expression = data
        else:
            self._expression = None

        self._analyzer = FluxAnalyzer(model=self._model,
                                      solver=solver,
                                      rxn_expr_score=self.expression)

    def __getattr__(self, item):
        return getattr(self._model, item)

    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, data):
        self._expression = Expression(self._model, data)

    @property
    def size(self):
        return 1

    @property
    def model(self):
        return self

    @property
    def cobra_model(self):
        return self._model

    @property
    def analyzer(self):
        return self._analyzer

    @property
    def reaction_ids(self):
        return [r.id for r in self._model.reactions]

    @property
    def gene_ids(self):
        return [g.id for g in self._model.genes]

    @property
    def metabolite_ids(self):
        return [m.id for m in self._model.metabolites]

    @property
    def subsystems(self):
        subs = {}
        for r in self._model.reactions:
            if r.subsystem in subs:
                subs[r.subsystem].append(r.id)
            else:
                subs[r.subsystem] = [r.id]
        return subs

    def do_analysis(self, **kwargs):
        self._analyzer.do_analysis(**kwargs)

    def get_flux(self, as_dict=False, **kwargs) -> Union[dict, pd.DataFrame]:
        flux_df = self._analyzer.get_flux(**kwargs)
        if not as_dict:
            return flux_df
        return {c: flux_df[c].to_frame().rename(columns={c: self.name_tag}) for c in flux_df.columns}

    def get_sol(self, **kwargs):
        return self._analyzer.get_sol(**kwargs)

    def save_analysis(self, file_dir_path):
        path = Path(file_dir_path)
        path.mkdir(parents=True, exist_ok=True)
        self._analyzer.save_analysis(file_path=path)

    def load_analysis(self, file_dir_path):
        path = Path(file_dir_path)
        self._analyzer.load_analysis(path)

    def get_problem_fluxes(self) -> pd.DataFrame:
        index, vals = [], []
        for var_name, val in self._model.solver.primal_values.items():
            index.append(var_name)
            vals.append(val)
        return pd.DataFrame({"fluxes": vals}, index=index)