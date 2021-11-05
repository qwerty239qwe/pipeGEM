from typing import Optional, Sequence, Union

import numpy as np
import cobra
from optlang.symbolics import Zero


class Problem:
    CSENSE = {"E": "equals to", "L": "lower than", "G": "greater than"}
    var_type_dict = {"C": "continuous", "B": "binary", "I": "integer"}

    def __init__(self, model = None):
        self._model = model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.prepare_problem()

    def _check_matrix(self, S, v_lbs, v_ubs, b, csense, objs, var_types, col_names=None, row_names=None):
        assert col_names is None or len(col_names) == S.shape[1], "number of column names does not match S.shape[1]"
        assert row_names is None or len(row_names) == S.shape[0], "number of row names does not match S.shape[0]"
        assert var_types is None or len(var_types) == S.shape[1], "number of var types does not match S.shape[1]"
        assert all([v in self.var_type_dict for v in var_types]), \
            f"all of the var types should be one of the {self.var_type_dict.keys()}"
        assert S.shape[1] == len(v_lbs) == len(v_ubs) == len(objs), \
            "number of v's lower bounds, number of upper bounds or the len of c vector does not equal to S.shape[1]"
        assert S.shape[0] == len(csense) == len(b), \
            "number of csense or the len of b vector does not equal to S.shape[0]"
        assert all(
            [c in self.CSENSE for c in csense]), \
            f"All of the csense should be one of the: {self.CSENSE.keys()}"

    def prepare_problem(self):
        self.S = cobra.util.create_stoichiometric_matrix(self.model)
        self.ori_lbs = np.array([r.lower_bound for r in self.model.reactions])
        self.ori_ubs = np.array([r.upper_bound for r in self.model.reactions])

        self.ori_b = np.zeros(len(self.model.reactions))
        self.ori_c = ["E" for _ in range(len(self.model.reactions))]
        self.ori_objs = np.array([r.objective_coefficient for r in self.model.reactions])

    def setup_problem(self,
                      S: np.ndarray,
                      v_lbs: Union[np.ndarray, Sequence[Union[float, int]]],
                      v_ubs: Union[np.ndarray, Sequence[Union[float, int]]],
                      b: Union[np.ndarray, Sequence[Union[float, int]]],
                      csense: Union[np.ndarray, Sequence[str]],
                      objs: Union[np.ndarray, Sequence[Union[float, int]]],
                      var_types: Union[np.ndarray, Sequence[Union[float, int]]] = None,
                      col_names: Union[np.ndarray, Sequence[str]] = None,
                      row_names: Union[np.ndarray, Sequence[str]] = None):
        """

        Parameters
        ----------
        S
        v_lbs: must be equal to S.shape[1]
        v_ubs: must be equal to S.shape[1]
        b
        csense: must be equal to S.shape[0]
        objs: must be equal to S.shape[1]
        var_types: must be equal to S.shape[1]
        col_names
        row_names

        Returns
        -------

        """

        self._check_matrix(S, v_lbs, v_ubs, b, csense, objs, var_types, col_names, row_names)
        new_model = cobra.Model()
        prob = new_model.problem
        variables = []
        constraints = []
        if var_types is None:
            var_types = ["C" for _ in range(S.shape[1])]

        for i, (v_lb, v_ub, v_type) in enumerate(zip(v_lbs, v_ubs, var_types)):
            m = prob.Variable(name=f"var_{i}" if col_names is None else col_names[i],
                              type=self.var_type_dict[v_type],
                              lb=v_lb,
                              ub=v_ub)
            variables.append(m)

        objs = {variables[i]: c for i, c in enumerate(objs)}

        for i, (b_bound, b_type) in enumerate(zip(b, csense)):
            non_zero_idx = np.nonzero(S[i])[0]
            constr_data = {"name": f"const_{i}" if row_names is None else row_names[i]}
            if b_type in ("E", "G"):
                constr_data.update({"lb": b_bound})
            if b_type in ("E", "L"):
                constr_data.update({"ub": b_bound})
            if b_bound != 0:
                n = prob.Constraint(
                    sum([float(S[i][j]) * variables[i] for j in non_zero_idx]),
                    **constr_data
                )
            else:
                n = prob.Constraint(
                    Zero,
                    **constr_data
                )
            constraints.append(n)

        new_model.add_cons_vars(variables + constraints)
        new_model.objective = prob.Objective(Zero, sloppy=True)
        new_model.objective.set_linear_coefficients(objs)
        return new_model