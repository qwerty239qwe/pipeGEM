from typing import Optional, Sequence, Union

import numpy as np
import cobra
from optlang.symbolics import Zero

from . import Model


class DimensionMismatchedError(ValueError):
    def __init__(self, *args):
        super(DimensionMismatchedError, self).__init__(*args)


class Problem:
    CSENSE = {"E": "equals to", "L": "lower than", "G": "greater than"}
    var_type_dict = {"C": "continuous", "B": "binary", "I": "integer"}

    def __init__(self, model = None):
        if isinstance(model, Model):
            self._model: cobra.Model = model.cobra_model
        else:
            self._model: cobra.Model = model
        self.S, self.v, self.lbs, self.ubs, self.b, self.c, self.objs, self.col_names, self.row_names = \
            self.prepare_problem(model)

    def copy(self):
        new_prob = Problem(model=self._model)
        for prop in ["S", "v", "lbs", "ubs", "b", "c", "objs", "col_names", "row_names"]:
            setattr(new_prob, prop, getattr(self, prop))
        return new_prob

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if isinstance(model, Model):
            self._model = model.cobra_model
        else:
            self._model = model
        self.S, self.v, self.lbs, self.ubs, self.b, self.c, self.objs, self.col_names, self.row_names = \
            self.prepare_problem(model)

    def _check_matrix(self):
        assert self.col_names is None or len(self.col_names) == self.S.shape[1], \
            f"number of column names ({len(self.col_names)}) does not match S.shape[1] ({self.S.shape[1]})"
        assert self.row_names is None or len(self.row_names) == self.S.shape[0], \
            f"number of row names ({len(self.row_names)}) does not match S.shape[0] ({self.S.shape[0]})"
        assert self.v is None or len(self.v) == self.S.shape[1], \
            f"number of var types ({len(self.v)}) does not match S.shape[1] ({self.S.shape[1]})"
        if self.v is not None:
            assert all([v in self.var_type_dict for v in self.v]), \
                f"all of the var types should be one of the {self.var_type_dict.keys()}"
        assert self.S.shape[1] == len(self.lbs) == len(self.ubs) == len(self.objs), \
            f"number of v's lower bounds: {len(self.lbs)}, number of upper bounds: {len(self.ubs)} " \
            f"or the len of c vector: {len(self.c)} does not equal to S.shape[1] {self.S.shape[1]}"
        assert self.S.shape[0] == len(self.c) == len(self.b), \
            f"number of csense: {len(self.c)} or the len of b vector: {len(self.b)} does not equal to " \
            f"S.shape[0]: {self.S.shape[0]}"
        assert all(
            [c in self.CSENSE for c in self.c]), \
            f"All of the csense should be one of the: {self.CSENSE.keys()}"

    @staticmethod
    def _check_extend_horizontal(e_S, e_v, e_v_lb, e_v_ub, e_objs, e_names=None):
        if not (e_S.shape[1] == e_v.shape[0] == e_v_lb.shape[0] == e_v_ub.shape[0]):
            raise DimensionMismatchedError("")
        if e_names is not None and len(e_names) != e_v.shape[0]:
            raise DimensionMismatchedError("")
        if e_objs is not None and len(e_objs) != e_v.shape[0]:
            raise DimensionMismatchedError("")

    @staticmethod
    def _check_extend_vertical(e_S, e_b, e_c, e_names=None):
        if not (e_S.shape[0] == e_b.shape[0] == e_c.shape[0]):
            raise DimensionMismatchedError(f"S shape[0]: {e_S.shape[0]}, b shape: {e_b.shape[0]}, "
                                           f"c shape: {e_c.shape[0]} should be matched with each other")
        if e_names is not None and len(e_names) != e_b.shape[0]:
            raise DimensionMismatchedError("")
        if e_c is not None and len(e_c) != e_b.shape[0]:
            raise DimensionMismatchedError("")

    @staticmethod
    def prepare_problem(model):
        S = cobra.util.create_stoichiometric_matrix(model)
        v = np.array(["C" for _ in range(len(model.reactions))])
        lbs = np.array([r.lower_bound for r in model.reactions])
        ubs = np.array([r.upper_bound for r in model.reactions])
        b = np.zeros(len(model.metabolites))
        c = np.array(["E" for _ in range(len(model.metabolites))])
        objs = np.array([r.objective_coefficient for r in model.reactions])
        cols, rows = np.array([r.id for r in model.reactions]), np.array([m.id for m in model.metabolites])
        return S, v, lbs, ubs, b, c, objs, cols, rows

    def get_rev(self):
        return (self.lbs != 0) & (self.ubs != 0)

    def modify_problem(self) -> None:
        """
        The modification of all problem components, subclass of Problem should change this to create a new problem

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def extend_horizontal(self, e_S, e_v, e_v_lb, e_v_ub, e_objs=None, e_names=None, at_right=True):
        e_objs = e_objs if e_objs is not None else np.zeros(e_S.shape[1])
        e_names = e_names if e_names is not None else np.array(
            [f"var_{i}" for i in range(len(self.col_names), len(self.col_names) + e_S.shape[1])])
        self._check_extend_horizontal(e_S, e_v, e_v_lb, e_v_ub, e_objs, e_names)
        self.S = np.concatenate([self.S, e_S] if at_right else [e_S, self.S], axis=1)
        self.v = np.concatenate([self.v, e_v] if at_right else [e_v, self.v])
        self.lbs = np.concatenate([self.lbs, e_v_lb] if at_right else [e_v_lb, self.lbs])
        self.ubs = np.concatenate([self.ubs, e_v_ub] if at_right else [e_v_ub, self.ubs])
        self.objs = np.concatenate([self.objs, e_objs] if at_right else [e_objs, self.objs])
        self.col_names = np.concatenate([self.col_names, e_names] if at_right else [e_names, self.col_names])

    def extend_vertical(self, e_S, e_b, e_c=None, e_names=None, at_bottom=True):
        e_c = e_c if e_c is not None else np.array(["C" for _ in range(e_S.shape[0])])
        e_names = e_names if e_names is not None else np.array(
            [f"const_{i}" for i in range(len(self.row_names), len(self.row_names) + e_S.shape[0])])
        self._check_extend_vertical(e_S, e_b, e_c, e_names)
        self.S = np.concatenate([self.S, e_S], axis=0 if at_bottom else [e_S, self.S])
        self.b = np.concatenate([self.b, e_b] if at_bottom else [e_b, self.b])
        self.c = np.concatenate([self.c, e_c] if at_bottom else [e_c, self.c])
        self.row_names = np.concatenate([self.row_names, e_names] if at_bottom else [e_names, self.row_names])

    def to_model(self, name_tag):
        """
        Returns
        -------

        """
        self.modify_problem()
        S, v, v_lbs, v_ubs, b, csense = self.S, self.v, self.lbs, self.ubs, self.b, self.c
        objs, col_names, row_names = self.objs, self.col_names, self.row_names
        self._check_matrix()
        new_model = cobra.Model()
        prob = new_model.problem
        variables = []
        constraints = []
        if v is None:
            v = ["C" for _ in range(S.shape[1])]

        for i, (v_lb, v_ub, v_type) in enumerate(zip(v_lbs, v_ubs, v)):
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
        return Model(new_model, name_tag=name_tag)