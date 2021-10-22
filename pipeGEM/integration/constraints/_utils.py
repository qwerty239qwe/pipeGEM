from functools import partial, wraps

from .constraint import CONSTR_DICT, follow_up


def add_constraint(func=None):
    if func is None:
        return partial(add_constraint)

    @wraps(func)
    def _do_analysis(self, **kwargs):
        constr_name = kwargs["constr"] if "constr" in kwargs else "default"
        constr_follow_up = kwargs.pop("do_follow_up") if "do_follow_up" in kwargs else False

        assert constr_name in CONSTR_DICT, "Valid constr_name: " + ", ".join(list(CONSTR_DICT.keys()))
        model, exp_score = getattr(self, "model"), getattr(self, "rxn_expr_score")

        with model:
            if constr_name not in ["default", "None"]:
                constr_kws = {k: kwargs.pop(k) for k in CONSTR_DICT[constr_name][1] if k in kwargs}
                if "model" in CONSTR_DICT[constr_name][1]:
                    constr_kws["model"] = model
                if "rxn_expr_score" in CONSTR_DICT[constr_name][1]:
                    constr_kws["rxn_expr_score"] = exp_score
                constr_details = CONSTR_DICT[constr_name][0](**constr_kws)  # apply additional constraints
            else:
                constr_details = None
            func(self, **kwargs)
            sol_df = self.get_df(method=kwargs["method"], constr=kwargs["constr"])
            if constr_follow_up and constr_details is not None:
                return follow_up(constr_details, constr=constr_name, sol_df=sol_df)
            return constr_details
    return _do_analysis


def add_constraint_f(func=None):
    if func is None:
        return partial(add_constraint_f)

    @wraps(func)
    def _do_analysis(*arg, **kwargs):
        constr_name = kwargs.pop("constr") if "constr" in kwargs else "default"
        constr_follow_up = kwargs.pop("do_follow_up") if "do_follow_up" in kwargs else False
        assert constr_name in CONSTR_DICT, "Valid constr_name: " + ", ".join(list(CONSTR_DICT.keys()))
        model = kwargs.get("model") if "model" in kwargs else arg[0]

        with model:
            if constr_name not in ["default", "None"]:
                # setup constraints
                constr_kws = {k: kwargs.pop(k) for k in CONSTR_DICT[constr_name][1] if k in kwargs}
                if "model" in CONSTR_DICT[constr_name][1]:
                    constr_kws["model"] = model
                constr_details = CONSTR_DICT[constr_name][0](**constr_kws)  # apply additional constraints
            else:
                constr_details = None
            sol_df = func(*arg, **kwargs)
            if constr_follow_up and constr_details is not None:
                return follow_up(constr_details, constr=constr_name, sol_df=sol_df)
            return constr_details
    return _do_analysis