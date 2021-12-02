from typing import Optional, Sequence, Union

import numpy as np
import cobra


from . import Model


class DimensionMismatchedError(ValueError):
    def __init__(self, *args):
        super(DimensionMismatchedError, self).__init__(*args)


class Problem:
    CSENSE = {"E": "equals to", "L": "lower than", "G": "greater than"}
    var_type_dict = {"C": "continuous", "B": "binary", "I": "integer"}
    req_args = ("S",)

    def __init__(self, model = None, **kwargs):
        if model is None:
            self._model = model
            infered = self._infer_attr(**kwargs)
            for k, v in infered.items():
                setattr(self, k, v)
        else:
            if isinstance(model, Model):
                self._model: cobra.Model = model.cobra_model
            else:
                self._model: cobra.Model = model
            self.S, self.v, self.lbs, self.ubs, self.b, self.c, self.objs, self.col_names, self.row_names = \
                self.prepare_problem(model)
        self._check_matrix()
        self.anchor_m, self.anchor_n = self.S.shape
        self.modify_problem()

    def copy(self):
        new_prob = Problem(model=self._model)
        for prop in ["S", "v", "lbs", "ubs", "b", "c", "objs", "col_names", "row_names"]:
            setattr(new_prob, prop, getattr(self, prop))
        return new_prob

    def _to_slicer(self, s):
        if s == "n":
            return self.anchor_n
        elif s == "m":
            return self.anchor_m
        else:
            return int(s)

    def parse_s_shape(self, s_shape) -> dict:
        S = self.S.copy()
        reconstruct_attr = True
        if "m" in s_shape[1] or "n" in s_shape[0]:
            reconstruct_attr = False
            s_shape = s_shape[1], s_shape[0]
            S = S.T

        if "m" in s_shape[1] or "n" in s_shape[0]:
            raise ValueError("No such reshape option provided")

        row_indexer, col_indexer = np.array([False for _ in range(S.shape[0])]), \
                                   np.array([False for _ in range(S.shape[1])])
        for i, dim in enumerate(["m", "n"]):
            if s_shape[i] == dim:
                S = S[:self.anchor_m, :] if i == 0 else S[:, :self.anchor_n]
                if i == 0 and reconstruct_attr:
                    row_indexer[:self.anchor_m] = True
                elif i == 1 and reconstruct_attr:
                    col_indexer[:self.anchor_n] = True
            elif ":" in s_shape[i]:
                if ":" == s_shape[i]:
                    pass
                slicer = s_shape[i].split(":")
                if len(slicer) == 2:
                    (s, t), i = [self._to_slicer(s) for s in slicer], 1
                else:
                    s, t, i = [self._to_slicer(s) for s in slicer]
                S = S[s:i:t, :] if i == 0 else S[:, s:i:t]
                if i == 0 and reconstruct_attr:
                    row_indexer[s:i:t] = True
                elif i == 1 and reconstruct_attr:
                    col_indexer[s:i:t] = True
            elif "," in s_shape[i]:
                index = [self._to_slicer(s) for s in s_shape[i].split(",")]
                S = S[index, :] if i == 0 else S[:, index]
                if i == 0 and reconstruct_attr:
                    row_indexer[index] = True
                elif i == 1 and reconstruct_attr:
                    col_indexer[index] = True

        if reconstruct_attr:
            v, b = self.v.copy()[col_indexer], self.b.copy()[row_indexer]
            lbs, ubs = self.lbs.copy()[col_indexer], self.ubs.copy()[col_indexer]
            c, objs = self.c.copy()[row_indexer], self.objs.copy()[col_indexer]
            col_names, row_names = self.col_names.copy()[col_indexer], self.row_names.copy()[row_indexer]
            return {"S": S, "v": v, "b": b, "c": c, "lbs": lbs, "ubs": ubs, "objs": objs,
                    "col_names": col_names, "row_names": row_names}
        return {"S": S}

    @classmethod
    def from_problem(cls, old_problem, s_shape=("m", "n"), **kwargs):
        elements = old_problem.parse_s_shape(s_shape)
        kwargs.update({"model": None})
        return cls(**dict(**elements, **kwargs))

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

    def _infer_attr(self, **kwargs):
        dic = dict()
        dic.update(kwargs)
        if "S" not in dic:
            raise ValueError("")
        if "v" not in dic:
            dic["v"] = np.array(["C" for _ in range(dic["S"].shape[1])])
        if "c" not in dic:
            dic["c"] = np.array(["E" for _ in range(dic["S"].shape[0])])
        if "lbs" not in dic:
            dic["lbs"] = np.zeros(dic["S"].shape[1])
        if "ubs" not in dic:
            dic["ubs"] = np.ones(dic["S"].shape[1])
        if "objs" not in dic:
            dic["objs"] = np.zeros(dic["S"].shape[1])
        if "b" not in dic:
            dic["b"] = np.zeros(dic["S"].shape[0])
        if "col_names" not in dic:
            dic["col_names"] = np.array([f"r_{i}" for i in range(dic["S"].shape[1])])
        if "row_names" not in dic:
            dic["row_names"] = np.array([f"m_{i}" for i in range(dic["S"].shape[0])])
        return dic

    def get_rev(self):
        return (self.lbs < 0) & (self.ubs > 0)

    def modify_problem(self) -> None:
        """
        The modification of all problem components,
        subclass of Problem should change this to create a new problem

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
        e_c = e_c if e_c is not None else np.array(["E" for _ in range(e_S.shape[0])])
        e_names = e_names if e_names is not None else np.array(
            [f"const_{i}" for i in range(len(self.row_names), len(self.row_names) + e_S.shape[0])])
        self._check_extend_vertical(e_S, e_b, e_c, e_names)
        self.S = np.concatenate([self.S, e_S], axis=0 if at_bottom else [e_S, self.S])
        self.b = np.concatenate([self.b, e_b] if at_bottom else [e_b, self.b])
        self.c = np.concatenate([self.c, e_c] if at_bottom else [e_c, self.c])
        self.row_names = np.concatenate([self.row_names, e_names] if at_bottom else [e_names, self.row_names])


class LP_Problem(Problem):
    def __init__(self, model, **kwargs):
        super().__init__(model = model, **kwargs)

    def modify_problem(self) -> None:
        pass