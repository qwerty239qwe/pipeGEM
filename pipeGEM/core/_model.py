from pathlib import Path

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
        self._lvl = 0
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
    def model(self):
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

    def get_analysis(self, method, constr="default", keep_rc=False):
        return self._analyzer.get_df(method=method, constr=constr, keep_rc=keep_rc)

    def save_analysis(self, file_dir_path):
        path = Path(file_dir_path)
        path.mkdir(parents=True, exist_ok=True)
        self._analyzer.save_analysis(file_path=path)

    def load_analysis(self, file_dir_path):
        path = Path(file_dir_path)
        self._analyzer.load_analysis(path)