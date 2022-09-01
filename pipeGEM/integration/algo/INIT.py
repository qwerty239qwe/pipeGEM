import cobra
import numpy as np
from optlang.symbolics import Zero
from collections import namedtuple


def add_fake_mets_for_core_mets(model, met_ids, added_comp="c"):
    # create fake met for essential mets
    mets_in_mod = [m.id for m in model.metabolites]
    new_mets = []
    for met_id in met_ids:
        if met_id in mets_in_mod:
            pm: cobra.Metabolite = model.metabolites.get_by_id(met_id)
            new_mets.append(cobra.Metabolite(id = "fake_PM_"+met_id,
                                             formula=pm.formula,
                                             name="fake_PM_"+pm.name,
                                             charge=pm.charge,
                                             compartment=added_comp
                                             ))
    return new_mets


def add_fake_mets_for_non_core_rxns(model, non_cores):
    rxns_in_mod = [r.id for r in model.reactions]
    assert all([nc in rxns_in_mod for nc in non_cores])
    new_mets, new_rxns = [], []
    for r in non_cores:
        fm = cobra.Metabolite(id = f"fake_met_for_{r}")
        fr = cobra.Reaction(id = f"fake_rxn_for_{r}", lower_bound=0, upper_bound=1)
        model.reactions.get_by_id(r).add_metabolites({fm: 1})
        new_mets.append(fm)
        new_rxns.append(fr)
    model.add_reactions(new_rxns)
    return new_mets, new_rxns


def add_sinks_for_mets(model, real_met_ids):
    mets_in_mod = [m.id for m in model.metabolites]
    new_rxns = []
    for m in real_met_ids:
        fr = cobra.Reaction(id = f"sink_for_{m}", lower_bound=0, upper_bound=1)
        new_rxns.append(fr)
    model.add_reactions(new_rxns)
    for mid, r in zip(real_met_ids, new_rxns):
        r.add_metabolites({model.metabolites.get_by_id(mid): -1})
    return new_rxns


def add_constr_for_rev_rxns(model, forward_rxn_ids, reverse_rxn_ids):
    # the forward_rxn_ids and the reverse_rxn_ids should be paired
    for fr, rr in zip(forward_rxn_ids, reverse_rxn_ids):
        im_f = cobra.Metabolite(f"indicator_met_for_{fr}")
        model.reactions.get_by_id(fr).add_metabolites({im_f: 1})
        im_r = cobra.Metabolite(f"indicator_met_for_{rr}")
        model.reactions.get_by_id(rr).add_metabolites({im_r: 1})
        ir = cobra.Reaction(f"indicator_rxn_for_{fr}")


def build_INIT_problem(model, ):
    pass

# ================== ^ old algo ^ ==================


ftINIT_step = namedtuple("ftINIT_step", ["pos_rev_off", "met_secretion", "prev_usage",
                                         "ignored_rxn_types", "ignored_mets",
                                         "MILP_params", "abs_mip_gap"])


class ftINIT_builder:
    _rt = np.array(["exchanges", "import", "transport", "advanced_transport",
                    "spontaneous_rxn_ids", "external", "custom_ignored_rxn_ids", "without_GPR"])
    _method = {"1+1": [ftINIT_step(pos_rev_off=False, met_secretion=False, prev_usage="ignore",
                                   ignored_rxn_types=_rt[:7], ignored_mets=[],
                                   MILP_params={"mipgap": 4e-4, "timelimit": 150}, abs_mip_gap=10),
                       ftINIT_step(pos_rev_off=False, met_secretion=False, prev_usage="ignore",
                                   ignored_rxn_types=_rt[[0, 4]], ignored_mets=[],
                                   MILP_params={"mipgap": 0.003, "timelimit": 5000}, abs_mip_gap=10)
                       ]}

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
            coefs = [v for _, v in r.metabolites.values()]
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
        return result_dic

    @staticmethod
    def _add_cons_to_model(model, var_coefs, prefix, lb, ub, ):
        con_dict = {}
        for name, coefs in var_coefs.items():
            cons = model.Constraint(Zero, name=f"{prefix}{name}", lb=lb, ub=ub)
            con_dict[f"{prefix}{name}"] = coefs
        model.add_cons_vars(list(con_dict.values()), sloppy=True)
        for name, coefs in con_dict.items():
            model.constraints[name].set_linear_coefficients(coefs)

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
            {"comps": rxn_groups["pos_rev_rxns"], "prefix": "PR_3_", "lb": 0, "ub": np.inf, "type": "b"},
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
        constrs = {r.id: {var_dict[key.format(r=r.id)]: coef for key, coef in coefs.items()} for r in rxns}
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
        return var.startswith(prefixes)


    def _init_constraints_details(self, var_dict, rxn_groups, mets, met_prod_rxns):
        pi_var_coefs = self._const_detail_helper(var_dict, coefs={"rxn_f_{r}": 1,
                                                                  "rxn_r_{r}": -1,
                                                                  "nY_pi_1_{r}": -0.1,
                                                                  "PI_{r}": -1,}, rxns=rxn_groups["pos_irrev_rxns"])

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

        pr4_var_coefs = self._const_detail_helper(var_dict, coefs={"PR_3_{r}": 1,
                                                                   "PR_4_{r}": 100,
                                                                   "PR_6_{r}": 1}, rxns=rxn_groups["pos_rev_rxns"])

        ni_var_coefs = self._const_detail_helper(var_dict, coefs={"rxn_f_{r}": 1,
                                                                  "rxn_r_{r}": -1,
                                                                  "nY_ni_1_{r}": -100,
                                                                  "NI_{r}": 1}, rxns=rxn_groups["neg_irrev_rxns"])

        nr1_var_coefs = self._const_detail_helper(var_dict, coefs={"rxn_f_{r}": 1,
                                                                   "rxn_r_{r}": -1,
                                                                   "NR_1_{r}": -1,
                                                                   "NR_2_{r}": 1}, rxns=rxn_groups["neg_rev_rxns"])

        nr2_var_coefs = self._const_detail_helper(var_dict, coefs={"nY_nr_1_{r}": 100,
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
                      mets
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

        for r in self.model.reactions:
            var_dict[f"rxn_f_{r.id}"] = r.forward_variable
            var_dict[f"rxn_r_{r.id}"] = r.reverse_variable

        met_prod_rxns = {m: self.get_met_prod_rxns(self.model, m) for m in mets}
        constr_add_details = self._init_constraints_details(var_dict, rxn_groups,
                                                            mets=mets, met_prod_rxns=met_prod_rxns)

        for c in constr_add_details:
            self._add_cons_to_model(model=self.model,
                                    var_coefs=c["var_coefs"],
                                    prefix=c["prefix"],
                                    lb=c["lb"], ub=c["ub"])
        obj_dict = {}
        for name, var in var_dict.items():
            if "nY_" == name[:3]:
                obj_dict[var] = rxn_score_dict[name[8:]]
            elif "met_1_" == name[:6]:
                obj_dict[var] = self.prod_weight

        self.model.objective = obj_dict
        self.model.solver.problem.setParam('MIPGap', self._mipgap)
        self.model.solver.problem.setParam('Timelimit', self._timelimit)

    def get_result(self,
                   rxn_score_dict,
                   essential_rxns,
                   mets
                   ):
        rev_rxns = [r.id for r in self.model.reactions if r.reversibility]
        ignored_rxns, reversed_rxns = [], []
        self.build_problem(rxn_score_dict, essential_rxns, ignored_rxns, mets)
        _ = self.model.solver.optimize()
        sol = self.model.solver.primal_values
        forward_fluxes = {varname: flux for varname, flux in sol.items() if not self.match_varname(varname,
                                                                                                   ["rxn_f_"])}
        reverse_fluxes = {varname: flux for varname, flux in sol.items() if not self.match_varname(varname,
                                                                                                   ["rxn_r_"])}
        fluxes = {varname[6:]: f - reverse_fluxes[varname[6:]] for varname, f in forward_fluxes.items()}

        rxns_turned_on = [varname[8:] for varname, flux in sol.items() if self.match_varname(varname,
                                                                                             ["nY_pi_1_",
                                                                                              "nY_pr_1_",
                                                                                              "nY_ni_1_",
                                                                                              "nY_nr_1_"]) and
                          flux >= 0.5]
        if self._prev_usage == "exclude":
            ignored_rxns += rxns_turned_on
        elif self._prev_usage == "essential":
            neg_flux = [r_id for r_id, f in fluxes.items() if f < 0]
            to_reverse = list(set(rxns_turned_on) & set(neg_flux) & set(rev_rxns))
            for r in to_reverse:
                rxn = self.model.reactions.get_by_id(r)
                rxn.subtract_metabolites({m: 2*c for m, c in rxn.metabolites.items()})
                rxn.upper_bound = -rxn.lower_bound
                rxn.lower_bound = 0
            essential_rxns += to_reverse
        elif self._prev_usage == "ignore":
            pass
        # check if the rxns



def apply_ftINIT(model,
                 data):
    model # simplify model
    rxn_score = data # transform to

    builder = ftINIT_builder(model=model)
