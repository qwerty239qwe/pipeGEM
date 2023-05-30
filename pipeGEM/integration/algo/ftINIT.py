import numpy as np
from optlang.symbolics import Zero
from collections import namedtuple
from functools import reduce

from pipeGEM.utils import flip_direction, get_rxn_set



ftINIT_step = namedtuple("ftINIT_step", ["pos_rev_off", "met_secretion", "prev_usage",
                                         "ignored_rxn_types", "ignored_mets",
                                         "MILP_params"])


class ftINIT_builder:
    _rt = np.array(["exchanges", "import", "transport", "advanced_transport",
                    "spontaneous_rxn_ids", "external", "custom_ignored_rxn_ids", "without_GPR"])
    _method_information = {"1+1": [ftINIT_step(pos_rev_off=False, met_secretion=False, prev_usage="ignore",
                                               ignored_rxn_types=_rt[:7], ignored_mets=[],
                                               MILP_params=[{"mipgap": 1, "timelimit": 150, "abs_mip_gap": 10},
                                                            {"mipgap": 3e-3, "timelimit": 5000, "abs_mip_gap": 20}]),
                                   ftINIT_step(pos_rev_off=False, met_secretion=False, prev_usage="essential",
                                               ignored_rxn_types=_rt[[0, 4]], ignored_mets=[],
                                               MILP_params=[{"mipgap": 4e-4, "timelimit": 5, "abs_mip_gap": 10},
                                                            {"mipgap": 4e-4, "timelimit": 150, "abs_mip_gap": 10},
                                                            {"mipgap": 3e-3, "timelimit": 5000, "abs_mip_gap": 20}])
                                   ]}
    onoff_var_prefixes = ["nY_pi_1_", "nY_pr_1_", "nY_ni_1_", "nY_nr_1_"]
    pos_var_prefixes = ["PI_"] + [f"PR_{i}_" for i in range(1, 7)]
    neg_var_prefixes = ["NI_"] + [f"NR_{i}_" for i in range(1, 4)]
    ess_var_prefixes = [f"ER_{i}_" for i in range(1, 7)]
    met_var_prefixes = [f"met_{i}_" for i in range(1, 8)]
    all_new_var_prefixes = onoff_var_prefixes + pos_var_prefixes + neg_var_prefixes + ess_var_prefixes + met_var_prefixes

    def __init__(self,
                 model,
                 spontaneous_rxn_ids=None,
                 to_ignore_rxn_ids=None,
                 method="1+1",
                 mipgap=0.003,
                 timelimit=5000):
        self.model = model
        self.force_on_lim, self.prod_weight = 0.1, 0.5
        self._mipgap = mipgap
        self._timelimit = timelimit
        self._method = method
        self._external_comp = 'e'
        if self._method not in ["1+1", "2+1", "2+0"]:
            raise ValueError("Invalid method")
        self._spontaneous_rxn_ids = spontaneous_rxn_ids if spontaneous_rxn_ids is not None else []
        self._to_ignore_rxn_ids = to_ignore_rxn_ids if to_ignore_rxn_ids is not None else []
        self._ignore_rxn_ids = self._get_ignored_rxn_ids()

    def _get_ignored_rxn_ids(self) -> dict:
        rxn_id_dic = {}
        rxn_id_dic["spontaneous_rxn_ids"] = self._spontaneous_rxn_ids
        rxn_id_dic["custom_ignored_rxn_ids"] = self._to_ignore_rxn_ids
        rxn_id_dic["exchanges"] = [r.id for r in self.model.exchanges]
        rxn_id_dic["import"] = []
        rxn_id_dic["transport"] = []
        rxn_id_dic["advanced_transport"] = []
        rxn_id_dic["external"] = []
        rxn_id_dic["without_GPR"] = []
        for r in self.model.reactions:
            coefs = [v for _, v in r.metabolites.items()]
            if len(r.metabolites) == 2 and (coefs[0] == -coefs[1]):
                comps = [m.compartment for m, v in r.metabolites.items()]
                met_ids = [m.id[:-1] for m, v in r.metabolites.items()]
                if not (len(set(met_ids)) == 1): # should be all the same
                    continue
                if any([c == self._external_comp for c in comps]):
                    rxn_id_dic["import"].append(r.id)
                else:
                    rxn_id_dic["transport"].append(r.id)
            elif r.gene_reaction_rule == '' and len(r.metabolites) > 2 and (len(r.metabolites) % 2 == 0) and sum(coefs) == 0:
                comps = [m.compartment for m, v in r.metabolites.items()]
                met_ids = [m.id[:-1] for m, v in r.metabolites.items()]
                if len(set(met_ids)) != (len(met_ids) // 2):
                    continue
                failed = False
                for one_met_id in set(met_ids):
                    matched_index = [i for i, x in enumerate(met_ids) if x == one_met_id]
                    if len(matched_index) > 2 or set([comps[i] for i in matched_index]) or sum([coefs[i] for i in matched_index]) != 0:
                        failed = True
                if failed:
                    continue
                rxn_id_dic["advanced_transport"].append(r.id)
            elif r.gene_reaction_rule == '' and len(r.metabolites) > 1 and all([m.compartment == self._external_comp for m in r.metabolites]):
                rxn_id_dic["external"].append(r.id)
            elif r.gene_reaction_rule == '':
                rxn_id_dic["without_GPR"].append(r.id)
        return rxn_id_dic

    def categorize_rxns(self,
                        model,
                        rxn_score_dict,
                        essential_rxns,
                        ignored_rxns,
                        mets) -> dict:
        result_dict = {}
        result_dict["rev_rxns"] = [r.id for r in model.reactions if r.reversibility]
        result_dict["irrev_rxns"] = [r.id for r in model.reactions if not r.reversibility]
        result_dict["ess_rev_rxns"] = list(set(essential_rxns) & set(result_dict["rev_rxns"]))
        result_dict["ess_irrev_rxns"] = list(set(essential_rxns) & set(result_dict["irrev_rxns"]))
        ess_related_mets = [m.id for r in essential_rxns for m in model.reactions.get_by_id(r).metabolites]
        mets = list(set(mets) - set(ess_related_mets))
        result_dict["mets_not_associated_w_ess"] = mets
        result_dict["met_rxns"] = list(
            set([r.id for m in mets for r in self.get_met_prod_rxns(model, m)]) & set(result_dict["irrev_rxns"]))
        result_dict["pos_rxns"] = set(
            [r for r, c in rxn_score_dict.items() if (c > 0 or ((c == 0 or (r in ignored_rxns))
                                                                and r in result_dict["met_rxns"]))]) - set(
            essential_rxns)
        result_dict["neg_rxns"] = set([r for r, c in rxn_score_dict.items() if c < 0]) - set(essential_rxns)
        result_dict["pos_rev_rxns"] = list(set(result_dict["pos_rxns"]) & set(result_dict["rev_rxns"]))
        result_dict["pos_irrev_rxns"] = list(set(result_dict["pos_rxns"]) & set(result_dict["irrev_rxns"]))
        result_dict["neg_rev_rxns"] = list(set(result_dict["neg_rxns"]) & set(result_dict["rev_rxns"]))
        result_dict["neg_irrev_rxns"] = list(set(result_dict["neg_rxns"]) & set(result_dict["irrev_rxns"]))
        if not (len(mets) == 0):
            result_dict["met_neg_rev_rxns"] = result_dict["neg_rev_rxns"] & result_dict["met_rxns"]
            result_dict["met_neg_irrev_rxns"] = result_dict["neg_irrev_rxns"] & result_dict["met_rxns"]

        return result_dict

    @staticmethod
    def get_met_prod_rxns(model,
                          met_id):
        met = model.metabolites.get_by_id(met_id)
        prod_rxns = []
        for r in met.reactions:
            if (r.metabolites[met] > 0 and r.lower_bound < 0) or (r.metabolites[met] < 0 and r.upper_bound > 0):
                prod_rxns.append(r.id)
        return prod_rxns

    @staticmethod
    def _build_vars_dict(model, rxn_ids, prefix, lb, ub, type_ab="c"):
        type_dic = {"c": "continuous", "b": "binary", "i": "integer"}
        result_dic = {}
        for r in rxn_ids:
            var = model.problem.Variable(f"{prefix}{r}", lb=lb, ub=ub, type=type_dic[type_ab])
            result_dic[f"{prefix}{r}"] = var
        model.add_cons_vars(list(result_dic.values()), sloppy=True)
        model.solver.update()
        return result_dic

    @staticmethod
    def _add_cons_to_model(model, var_coefs, prefix, lb, ub, ):
        con_dict = {}
        added_cons = []
        for name, coefs in var_coefs.items():
            cons = model.problem.Constraint(Zero, name=f"{prefix}{name}", lb=lb, ub=ub)
            con_dict[f"{prefix}{name}"] = coefs
            added_cons.append(cons)
        model.add_cons_vars(added_cons, sloppy=True)
        for name, coefs in con_dict.items():
            model.constraints[name].set_linear_coefficients(coefs)
        model.solver.update()

    @staticmethod
    def _init_var_details(rxn_groups, mets, force_on_lim=0.1):

        var_add_patterns = [
            {"comps": rxn_groups["pos_irrev_rxns"], "prefix": "nY_pi_1_", "lb": 0, "ub": 1, "type": "c"},
            {"comps": rxn_groups["pos_rev_rxns"], "prefix": "nY_pr_1_", "lb": 0, "ub": 1, "type": "c"},
            {"comps": rxn_groups["neg_irrev_rxns"], "prefix": "nY_ni_1_", "lb": 0, "ub": 1, "type": "b"},
            {"comps": rxn_groups["neg_rev_rxns"], "prefix": "nY_nr_1_", "lb": 0, "ub": 1, "type": "b"},
            {"comps": rxn_groups["pos_irrev_rxns"], "prefix": "PI_", "lb": 0, "ub": np.inf, "type": "c"},
            {"comps": rxn_groups["pos_rev_rxns"], "prefix": "PR_1_", "lb": 0, "ub": np.inf, "type": "c"},
            {"comps": rxn_groups["pos_rev_rxns"], "prefix": "PR_2_", "lb": 0, "ub": np.inf, "type": "c"},
            {"comps": rxn_groups["pos_rev_rxns"], "prefix": "PR_3_", "lb": 0, "ub": 1, "type": "b"},
            {"comps": rxn_groups["pos_rev_rxns"], "prefix": "PR_4_", "lb": 0, "ub": np.inf, "type": "c"},
            {"comps": rxn_groups["pos_rev_rxns"], "prefix": "PR_5_", "lb": 0, "ub": np.inf, "type": "c"},
            {"comps": rxn_groups["pos_rev_rxns"], "prefix": "PR_6_", "lb": -100, "ub": np.inf, "type": "c"},
            {"comps": rxn_groups["neg_irrev_rxns"], "prefix": "NI_", "lb": 0, "ub": np.inf, "type": "c"},
            {"comps": rxn_groups["neg_rev_rxns"], "prefix": "NR_1_", "lb": 0, "ub": np.inf, "type": "c"},
            {"comps": rxn_groups["neg_rev_rxns"], "prefix": "NR_2_", "lb": 0, "ub": np.inf, "type": "c"},
            {"comps": rxn_groups["neg_rev_rxns"], "prefix": "NR_3_", "lb": 0, "ub": np.inf, "type": "c"},
            {"comps": rxn_groups["ess_rev_rxns"], "prefix": "ER_1_", "lb": 0, "ub": np.inf, "type": "c"},
            {"comps": rxn_groups["ess_rev_rxns"], "prefix": "ER_2_", "lb": 0, "ub": np.inf, "type": "c"},
            {"comps": rxn_groups["ess_rev_rxns"], "prefix": "ER_3_", "lb": force_on_lim, "ub": np.inf, "type": "c"},
            {"comps": rxn_groups["ess_rev_rxns"], "prefix": "ER_4_", "lb": 0, "ub": 1, "type": "b"},
            {"comps": rxn_groups["ess_rev_rxns"], "prefix": "ER_5_", "lb": 0, "ub": np.inf, "type": "c"},
            {"comps": rxn_groups["ess_rev_rxns"], "prefix": "ER_6_", "lb": -100, "ub": np.inf, "type": "c"},
            ]
        if len(rxn_groups["met_rxns"]) != 0:
            var_add_patterns.extend([{"comps": mets, "prefix": "met_1_", "lb": 0, "ub": 1, "type": "c"},
                                     {"comps": mets, "prefix": "met_2_", "lb": 0, "ub": np.inf, "type": "c"},
                                     {"comps": rxn_groups["met_neg_rev_rxns"], "prefix": "met_3_", "lb": 0, "ub": 1,
                                      "type": "b"},
                                     {"comps": rxn_groups["met_neg_rev_rxns"], "prefix": "met_4_", "lb": 0,
                                      "ub": np.inf, "type": "c"},
                                     {"comps": rxn_groups["met_neg_rev_rxns"], "prefix": "met_5_", "lb": -100,
                                      "ub": np.inf, "type": "c"},
                                     {"comps": rxn_groups["met_neg_rev_rxns"], "prefix": "met_6_", "lb": 0,
                                      "ub": np.inf, "type": "c"},
                                     {"comps": rxn_groups["met_neg_irrev_rxns"], "prefix": "met_7_", "lb": 0,
                                      "ub": np.inf, "type": "c"}])
        return var_add_patterns

    @staticmethod
    def _const_detail_helper(var_dict, coefs, rxns) -> dict:
        constrs = {r: {var_dict[key.format(r=r)]: coef for key, coef in coefs.items()} for r in rxns}
        return constrs

    @staticmethod
    def _search_met_prod_rxn_in_nY(var_dict, rxn_id):
        var_names = [v for v in var_dict if "nY_" == v[:3]]
        pi_vars = [v[8:] for v in var_names if "nY_pi_1_" in v]
        pr_vars = [v[8:] for v in var_names if "nY_pr_1_" in v]
        ni_vars = [v[8:] for v in var_names if "nY_ni_1_" in v]
        nr_vars = [v[8:] for v in var_names if "nY_nr_1_" in v]
        if not any([rxn_id in v for v in [pi_vars, pr_vars, nr_vars, ni_vars]]):
            raise ValueError("No met production reaction is found")
        for prefix, v in zip(["nY_pi_1_", "nY_pr_1_", "nY_nr_1_", "nY_ni_1_"], [pi_vars, pr_vars, nr_vars, ni_vars]):
            if rxn_id in v:
                return var_dict[f"{prefix}{rxn_id}"]

    def _constr_detail_for_met_var(self, var_dict, mets, met_prod_rxns):
        constrs = {}
        for met in mets:
            coefs = {self._search_met_prod_rxn_in_nY(var_dict, rxn_id): -1 for rxn_id in met_prod_rxns[met]}
            coefs.update({var_dict[f"met_1_{met}"]: 1, var_dict[f"met_2_{met}"]: 1})
            constrs[met] = coefs
        return constrs

    @staticmethod
    def match_varname(var:str, prefixes):
        if isinstance(prefixes, list):
            for i in prefixes:
                if var.startswith(i):
                    return True
            return False
        return var.startswith(prefixes)

    def _init_constraints_details(self, var_dict, rxn_groups, mets, met_prod_rxns):
        pi_var_coefs = self._const_detail_helper(var_dict, coefs={"rxn_f_{r}": 1,
                                                                  "rxn_r_{r}": -1,
                                                                  "nY_pi_1_{r}": -0.1,
                                                                  "PI_{r}": -1}, rxns=rxn_groups["pos_irrev_rxns"])

        pr1_var_coefs = self._const_detail_helper(var_dict, coefs={"rxn_f_{r}": 1,
                                                                   "rxn_r_{r}": -1,
                                                                   "PR_1_{r}": -1,
                                                                   "PR_2_{r}": 1}, rxns=rxn_groups["pos_rev_rxns"])

        pr2_var_coefs = self._const_detail_helper(var_dict, coefs={"nY_pr_1_{r}": -0.1,
                                                                   "PR_1_{r}": 1,
                                                                   "PR_2_{r}": 1,
                                                                   "PR_4_{r}": -1}, rxns=rxn_groups["pos_rev_rxns"])

        pr3_var_coefs = self._const_detail_helper(var_dict, coefs={"PR_1_{r}": 1,
                                                                   "PR_3_{r}": -100,
                                                                   "PR_5_{r}": 1}, rxns=rxn_groups["pos_rev_rxns"])

        pr4_var_coefs = self._const_detail_helper(var_dict, coefs={"PR_2_{r}": 1,
                                                                   "PR_3_{r}": 100,
                                                                   "PR_6_{r}": 1}, rxns=rxn_groups["pos_rev_rxns"])

        ni_var_coefs = self._const_detail_helper(var_dict, coefs={"rxn_f_{r}": 1,
                                                                  "rxn_r_{r}": -1,
                                                                  "nY_ni_1_{r}": -100,
                                                                  "NI_{r}": 1}, rxns=rxn_groups["neg_irrev_rxns"])

        nr1_var_coefs = self._const_detail_helper(var_dict, coefs={"rxn_f_{r}": 1,
                                                                   "rxn_r_{r}": -1,
                                                                   "NR_1_{r}": -1,
                                                                   "NR_2_{r}": 1}, rxns=rxn_groups["neg_rev_rxns"])

        nr2_var_coefs = self._const_detail_helper(var_dict, coefs={"nY_nr_1_{r}": -100,
                                                                   "NR_1_{r}": 1,
                                                                   "NR_2_{r}": 1,
                                                                   "NR_3_{r}": 1}, rxns=rxn_groups["neg_rev_rxns"])

        er1_var_coefs = self._const_detail_helper(var_dict, coefs={"rxn_f_{r}": 1,
                                                                   "rxn_r_{r}": -1,
                                                                   "ER_1_{r}": -1,
                                                                   "ER_2_{r}": 1}, rxns=rxn_groups["ess_rev_rxns"])

        er2_var_coefs = self._const_detail_helper(var_dict, coefs={"ER_1_{r}": 1,
                                                                   "ER_2_{r}": 1,
                                                                   "ER_3_{r}": -1}, rxns=rxn_groups["ess_rev_rxns"])

        er3_var_coefs = self._const_detail_helper(var_dict, coefs={"ER_1_{r}": 1,
                                                                   "ER_4_{r}": -100,
                                                                   "ER_5_{r}": 1}, rxns=rxn_groups["ess_rev_rxns"])

        er4_var_coefs = self._const_detail_helper(var_dict, coefs={"ER_2_{r}": 1,
                                                                   "ER_4_{r}": 100,
                                                                   "ER_6_{r}": 1}, rxns=rxn_groups["ess_rev_rxns"])
        constr_add_details = [{"prefix": "PI_", "var_coefs": pi_var_coefs, "lb": 0, "ub": 0},
                              {"prefix": "PR1_", "var_coefs": pr1_var_coefs, "lb": 0, "ub": 0},
                              {"prefix": "PR2_", "var_coefs": pr2_var_coefs, "lb": 0, "ub": 0},
                              {"prefix": "PR3_", "var_coefs": pr3_var_coefs, "lb": 0, "ub": 0},
                              {"prefix": "PR4_", "var_coefs": pr4_var_coefs, "lb": 0, "ub": 0},
                              {"prefix": "NI_", "var_coefs": ni_var_coefs, "lb": 0, "ub": 0},
                              {"prefix": "NR1_", "var_coefs": nr1_var_coefs, "lb": 0, "ub": 0},
                              {"prefix": "NR2_", "var_coefs": nr2_var_coefs, "lb": 0, "ub": 0},
                              {"prefix": "ER1_", "var_coefs": er1_var_coefs, "lb": 0, "ub": 0},
                              {"prefix": "ER2_", "var_coefs": er2_var_coefs, "lb": 0, "ub": 0},
                              {"prefix": "ER3_", "var_coefs": er3_var_coefs, "lb": 0, "ub": 0},
                              {"prefix": "ER4_", "var_coefs": er4_var_coefs, "lb": 0, "ub": 0}
                              ]
        if mets is not None and len(mets) != 0:
            met_var_coefs = self._constr_detail_for_met_var(var_dict, mets, met_prod_rxns)

            met_nr1_var_coefs = self._const_detail_helper(var_dict, coefs={"NR_1_{r}": 1,
                                                                           "met_3_{r}": -100,
                                                                           "met_4_{r}": 1},
                                                          rxns=rxn_groups["met_neg_rev_rxns"])

            met_nr2_var_coefs = self._const_detail_helper(var_dict, coefs={"NR_2_{r}": 1,
                                                                           "met_3_{r}": 100,
                                                                           "met_5_{r}": 1},
                                                          rxns=rxn_groups["met_neg_rev_rxns"])

            met_nr3_var_coefs = self._const_detail_helper(var_dict, coefs={"NR_1_{r}": 1,
                                                                           "NR_2_{r}": 1,
                                                                           "nY_nr_1_{r}": -0.1,
                                                                           "met_6_{r}": -1},
                                                          rxns=rxn_groups["met_neg_rev_rxns"])

            met_nr4_var_coefs = self._const_detail_helper(var_dict, coefs={"rxn_f_{r}": 1,
                                                                           "rxn_r_{r}": -1,
                                                                           "nY_ni_1_{r}": -0.1,
                                                                           "met_7_{r}": -1},
                                                          rxns=rxn_groups["met_neg_rev_rxns"])

            constr_add_details += [{"prefix": "MET_", "var_coefs": met_var_coefs, "lb": 0, "ub": 0},
                                   {"prefix": "MET_NR1_", "var_coefs": met_nr1_var_coefs, "lb": 0, "ub": 0},
                                   {"prefix": "MET_NR2_", "var_coefs": met_nr2_var_coefs, "lb": 0, "ub": 0},
                                   {"prefix": "MET_NR3_", "var_coefs": met_nr3_var_coefs, "lb": 0, "ub": 0},
                                   {"prefix": "MET_NR4_", "var_coefs": met_nr4_var_coefs, "lb": 0, "ub": 0}
                                   ]

        return constr_add_details

    def build_problem(self,
                      rxn_score_dict,
                      essential_rxns,
                      ignored_rxns,
                      fluxes,
                      mets,
                      mipgap,
                      timelimit
                      ):
        rxn_groups = self.categorize_rxns(self.model,
                                          rxn_score_dict=rxn_score_dict,
                                          essential_rxns=essential_rxns,
                                          ignored_rxns=ignored_rxns,
                                          mets=mets)
        var_dict = {}
        var_add_details = self._init_var_details(rxn_groups=rxn_groups, mets=mets,
                                                 force_on_lim=self.force_on_lim)
        for pt in var_add_details:

            added_vars = self._build_vars_dict(self.model,
                                               pt["comps"],
                                               pt["prefix"],
                                               lb=pt["lb"], ub=pt["ub"],
                                               type_ab=pt["type"])
            var_dict.update(added_vars)
            print(f"{len(added_vars)} vars added")

        for r in self.model.reactions:
            var_dict[f"rxn_f_{r.id}"] = r.forward_variable
            var_dict[f"rxn_r_{r.id}"] = r.reverse_variable

        met_prod_rxns = {m: self.get_met_prod_rxns(self.model, m) for m in mets}
        constr_add_details = self._init_constraints_details(var_dict, rxn_groups,
                                                            mets=mets, met_prod_rxns=met_prod_rxns)
        print("after var inclusion: ", len(self.model.variables), len(self.model.constraints))
        for c in constr_add_details:
            self._add_cons_to_model(model=self.model,
                                    var_coefs=c["var_coefs"],
                                    prefix=c["prefix"],
                                    lb=c["lb"],
                                    ub=c["ub"])
        for r in rxn_groups["ess_irrev_rxns"]:
            if fluxes[r] > 0:
                self.model.reactions.get_by_id(r).lower_bound = min(fluxes[r] * 0.99, 0.1)
            print(f"force irrev essential rxn {r} to have this lower_bound: {self.model.reactions.get_by_id(r).lower_bound}")

        obj_dict = {}
        for name, var in var_dict.items():
            if "nY_" == name[:3]:
                obj_dict[var] = -rxn_score_dict[name[8:]]
            elif "met_1_" == name[:6]:
                obj_dict[var] = -self.prod_weight
        self.model.objective.set_linear_coefficients({v: 0 for v in self.model.variables})
        self.model.objective.set_linear_coefficients(obj_dict)
        self.model.solver.problem.setParam('MIPGap', mipgap)
        #self.model.solver.problem.setParam('Timelimit', timelimit)
        self.model.solver.update()

    @staticmethod
    def _log_step_info(step, i, previous_result=None):
        print(f"Starting step {i + 1}")
        print("ignoring rxn types: ", step.ignored_rxn_types)
        print("method to use the previous result: ", step.prev_usage)
        if previous_result is not None:
            print(previous_result)

    def get_result(self,
                   rxn_score_dict,
                   essential_rxns,
                   mets,
                   init_flux_value=0.01
                   ):
        backward_rxns = get_rxn_set(self.model, "backward")
        if len(backward_rxns) > 0:
            print(f"Found and flipped {len(backward_rxns)} reactions")
            flip_direction(self.model, backward_rxns)

        rxns_turned_on = set()
        to_reverse, fluxes = [], {r.id: init_flux_value for r in self.model.reactions}
        ignored_rxns = []
        original_n_rxns = len(self.model.reactions)
        for i, step in enumerate(self._method_information[self._method]):
            self._log_step_info(step, i)
            ignored_rxns = [set(self._ignore_rxn_ids[ig]) for ig in step.ignored_rxn_types]
            ignored_rxns = list(reduce(set.union, ignored_rxns))
            rev_rxns = [r.id for r in self.model.reactions if r.reversibility]

            if step.prev_usage == "exclude":
                ignored_rxns += rxns_turned_on
            elif step.prev_usage == "essential":
                neg_flux = [r_id for r_id, f in fluxes.items() if f < 0]
                to_reverse = list(set(rxns_turned_on) & set(neg_flux) & set(rev_rxns))
                for r in to_reverse:
                    rxn = self.model.reactions.get_by_id(r)
                    rxn.subtract_metabolites({m: 2 * c for m, c in rxn.metabolites.items()})
                    rxn.upper_bound = -rxn.lower_bound
                    rxn.lower_bound = 0
                essential_rxns += to_reverse
            elif step.prev_usage == "ignore":
                pass

            mingap_result = 1
            next_mingap = None
            print(len(self.model.variables), len(self.model.constraints))
            for i_pa, params in enumerate(step.MILP_params):
                print(f"iteration {i_pa}")
                with self.model:
                    for exh in self.model.exchanges:
                        if -1000 < exh.lower_bound < 0:
                            exh.lower_bound = -1000
                        if 0 < exh.upper_bound < -1000:
                            exh.upper_bound = 1000

                    status = "Undetermined"
                    target_mingap = params["mipgap"] if i_pa == 0 else next_mingap
                    self.build_problem(rxn_score_dict, essential_rxns, ignored_rxns,
                                       fluxes=fluxes,
                                       mets=mets,
                                       mipgap=target_mingap,
                                       timelimit=params["timelimit"])
                    print(len(self.model.variables), len(self.model.constraints))
                    while status != "optimal":
                        self.model.objective.direction = "min"
                        status = self.model.solver.optimize()
                        if status != "optimal":
                            print(len(self.model.variables), len(self.model.constraints))
                            break

                            #target_mingap *= 10
                            #self.model.solver.problem.setParam('MIPGap', target_mingap)
                        print(f"target_mingap: {target_mingap:.4f}; optimization status: {status}")
                    sol = self.model.solver.primal_values

                    forward_fluxes = {varname: flux for varname, flux in sol.items()
                                      if not self.match_varname(varname, self.all_new_var_prefixes) and "_reverse" not in varname}
                    reverse_fluxes = {varname[:varname.index("_reverse")]: flux for varname, flux in sol.items()
                                      if not self.match_varname(varname, self.all_new_var_prefixes) and "_reverse" in varname}
                    fluxes = {rxn_id: f - reverse_fluxes[rxn_id] if rxn_id not in to_reverse else reverse_fluxes[rxn_id] - f
                              for rxn_id, f in forward_fluxes.items()}
                    assert len(fluxes) == original_n_rxns, f"Some rxns ({original_n_rxns - len(fluxes)}) are not in the output fluxes"
                    rxns_turned_on |= set([varname[8:] for varname, flux in sol.items() if self.match_varname(varname,
                                                                                                         ["nY_pi_1_",
                                                                                                          "nY_pr_1_",
                                                                                                          "nY_ni_1_",
                                                                                                          "nY_nr_1_"]) and
                                          flux >= 0.5])

                    mingap_result = self.model.solver.problem.MIPGAP

                    if i_pa < len(step.MILP_params) - 1:
                        next_mingap = min(1, max(step.MILP_params[i_pa+1]["mipgap"],
                                          step.MILP_params[i_pa+1]["abs_mip_gap"] / abs(self.model.solver.problem.ObjVal)))

                    if mingap_result < next_mingap:
                        print(f"Found valid MINGAP in iteration {i_pa}, result MINGap: {mingap_result}")
                        break
                    print(f"current mingap: {mingap_result}")

        print("MILP finished. Removing reactions...")
        to_remove = set([r.id for r in self.model.reactions]) - set(essential_rxns + ignored_rxns + list(rxns_turned_on))
        self.model.remove_reactions(list(to_remove), remove_orphans=True)
        return self.model


def apply_ftINIT(model,
                 data,
                 essential_rxns,
                 essential_mets,
                 simplify_method="linear"):
    model # simplify model
    rxn_scores = data # transform to

    builder = ftINIT_builder(model=model)
    builder.get_result(rxn_scores, essential_rxns, essential_mets)
