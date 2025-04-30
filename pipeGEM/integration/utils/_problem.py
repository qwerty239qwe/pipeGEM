from optlang.symbolics import Zero
from optlang.exceptions import IndicatorConstraintsNotSupported


type_dic = {"c": "continuous", "b": "binary", "i": "integer"}


def build_vars_dict_from_rxns(model,
                              rxn_ids,
                              prefix,
                              lbs,
                              ubs,
                              type_ab="c"):

    result_dic = {}
    if not hasattr(lbs, "__len__"):
        lbs = [lbs for _ in rxn_ids]
    if not hasattr(ubs, "__len__"):
        ubs = [ubs for _ in rxn_ids]

    for r, lb, ub in zip(rxn_ids, lbs, ubs):
        var = model.problem.Variable(f"{prefix}{r}", lb=lb, ub=ub, type=type_dic[type_ab])
        result_dic[f"{prefix}{r}"] = var
    model.add_cons_vars(list(result_dic.values()), sloppy=True)
    model.solver.update()
    return result_dic


def add_cons_to_model(model,
                      var_coefs,
                      prefix,
                      lbs,
                      ubs,
                      binary_vars=None,
                      bin_active_val=1,
                      use_gurobi=True):
    con_dict = {}
    added_cons = []
    if not hasattr(lbs, "__len__"):
        lbs = [lbs for _ in var_coefs]
    if not hasattr(ubs, "__len__"):
        ubs = [ubs for _ in var_coefs]

    if binary_vars is None:
        for (name, coefs), lb, ub in zip(var_coefs.items(), lbs, ubs):
            cons = model.problem.Constraint(Zero, name=f"{prefix}{name}", lb=lb, ub=ub)
            con_dict[f"{prefix}{name}"] = coefs
            added_cons.append(cons)
        model.add_cons_vars(added_cons, sloppy=True)
    else:
        for (name, coefs), lb, ub, b_ind in zip(var_coefs.items(), lbs, ubs, binary_vars):
            if not use_gurobi:
                try:
                    cons = model.problem.Constraint(Zero, name=f"{prefix}{name}", lb=lb, ub=ub,
                                                    indicator_variable=b_ind,
                                                    active_when=bin_active_val)
                    con_dict[f"{prefix}{name}"] = coefs
                    added_cons.append(cons)
                except IndicatorConstraintsNotSupported:
                    print(f"Indicator constraint not supported in {prefix}{name}")
                    return False
            else:
                if lb == ub:
                    _gurobi_add_indicator(problem=model.solver.problem, var_coefs=coefs, name=name, bound=lb,
                                          bound_type="EQUAL", indicator_variable=b_ind, active_when=bin_active_val)
                else:
                    _gurobi_add_indicator(problem=model.solver.problem, var_coefs=coefs, name=name, bound=lb,
                                          bound_type="GREATER_EQUAL", indicator_variable=b_ind,
                                          active_when=bin_active_val)
                    _gurobi_add_indicator(problem=model.solver.problem, var_coefs=coefs, name=name, bound=ub,
                                          bound_type="LESS_EQUAL", indicator_variable=b_ind,
                                          active_when=bin_active_val)
        model.add_cons_vars(added_cons, sloppy=True)
    for name, coefs in con_dict.items():
        model.constraints[name].set_linear_coefficients(coefs)
        # print(coefs)
    model.solver.update()
    return True


def _gurobi_add_indicator(problem, name, bound, var_coefs, bound_type, indicator_variable, active_when):
    from gurobipy import GRB
    from gurobipy import quicksum

    problem.addGenConstrIndicator(problem.getVarByName(indicator_variable.name),
                                  binval=active_when,
                                  lhs=quicksum([problem.getVarByName(k.name) * v for k, v in var_coefs.items()]),
                                  sense=getattr(GRB, bound_type),
                                  rhs=bound,
                                  name=name)
    problem.update()