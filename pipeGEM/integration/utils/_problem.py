from optlang.symbolics import Zero


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
                      ubs):
    con_dict = {}
    added_cons = []
    if not hasattr(lbs, "__len__"):
        lbs = [lbs for _ in var_coefs]
    if not hasattr(ubs, "__len__"):
        ubs = [ubs for _ in var_coefs]

    for (name, coefs), lb, ub in (var_coefs.items(), lbs, ubs):
        cons = model.problem.Constraint(Zero, name=f"{prefix}{name}", lb=lb, ub=ub)
        con_dict[f"{prefix}{name}"] = coefs
        added_cons.append(cons)
    model.add_cons_vars(added_cons, sloppy=True)
    for name, coefs in con_dict.items():
        model.constraints[name].set_linear_coefficients(coefs)
    model.solver.update()