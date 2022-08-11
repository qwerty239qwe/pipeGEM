import cobra
import numpy as np
from decimal import Decimal


def calc_rxn_unbalance_scores(rxn):
    all_c = [abs(c) for m, c in rxn.metabolites.items()]
    return (np.log10(max(all_c) / min(all_c)),  # to detect unbalanced
            np.mean([np.log10(max(all_c) / c) for c in all_c]),  # how many are unbalanced
            np.mean([np.log10(c) for c in all_c]))  # differ from 0


def get_met_coef_in_rxns(met):
    all_c = {r.id: abs(r.metabolites[met]) for r in met.reactions}
    return all_c


def get_met_unbalance_scores_in_rxns(met):
    all_c = {r.id: abs(r.metabolites[met]) for r in met.reactions}
    return all_c


def calc_met_unbalance_scores(met):
    all_c = [abs(r.metabolites[met]) for r in met.reactions]
    return np.log10(max(all_c) / min(all_c))


def calc_met_mean_unbalance_scores(met):
    all_c = [abs(r.metabolites[met]) for r in met.reactions]
    return np.mean([np.log10(c) for c in all_c])


def calc_met_unbalance_scores_mean_in_rxns(met):
    all_c = [max([abs(c) for m, c in r.metabolites.items()]) / abs(r.metabolites[met]) for r in met.reactions]
    return np.mean(np.log10(np.array([c for c in all_c])))


def add_scale_rxn_for_met(mod, met_id, mod_rxns, c=0.1, scale_f=1e-4):
    cur_f = Decimal("1")
    c = Decimal(f"{c}")
    scale_f = Decimal(f"{scale_f}")
    scale_times = 0
    original_met_id = met_id
    original_met = mod.metabolites.get_by_id(original_met_id)
    met_ids_in_mod = [m.id for m in mod.metabolites]

    while (cur_f < (scale_f * Decimal("0.999")) and scale_f > 1) or (
            cur_f > scale_f * Decimal("1.0001") and scale_f < 1):
        scale_times += 1
        met_postfix = f"_{c}_toPower_{scale_times}"
        cur_f = cur_f * c
        met = mod.metabolites.get_by_id(met_id)
        if original_met_id + met_postfix in met_ids_in_mod:
            print(f"{original_met_id + met_postfix} is already in the model")
            continue

        new_met = cobra.Metabolite(id=original_met_id + met_postfix, name=original_met.name + f"_{cur_f}",
                                   compartment=original_met.compartment)
        new_met.elements = {k: v * float(c) for k, v in met.elements.items()}

        new_rxn = cobra.Reaction(id=f"{met_id}_scale_to_{cur_f}", lower_bound=-1000 * (float(1 / c) ** scale_times),
                                 upper_bound=1000 * (float(1 / c) ** scale_times))
        mod.add_reactions([new_rxn])
        new_rxn.add_metabolites({new_met: float(1 / c),
                                 met: -1})
        met_id = original_met_id + met_postfix

    for r in mod_rxns:
        rxn = mod.reactions.get_by_id(r)
        rxn.subtract_metabolites({original_met: rxn.metabolites[original_met],
                                  new_met: -float(Decimal(rxn.metabolites[original_met]) / cur_f)})


class MetScorer:
    def __init__(self, met_id, model):
        self.met_id = met_id
        self.met = model.metabolites.get_by_id(met_id)
        self.coef_in_rxns = get_met_coef_in_rxns(self.met)
        self.ub_mean_scores_in_rxns = calc_met_unbalance_scores_mean_in_rxns(self.met)
        self.ub_scores = calc_met_unbalance_scores(self.met)
        self.ub_mean_scores = calc_met_mean_unbalance_scores(self.met)

    @staticmethod
    def _check_scale(rxn_coefs, bin=10):
        log_s = {k: np.log10(v) / np.log10(bin) for k, v in rxn_coefs.items()}
        min_s = int(min(log_s.values()) // 1)
        max_s = int(max(log_s.values()) // 1 + 1)
        # naive
        bin_dic = {f"{i}": [] for i in range(min_s, max_s)}
        for k, s in log_s.items():
            bin_dic[f"{int(s // 1)}"].append(k)

        return {k: v for k, v in bin_dic.items() if len(v) != 0}

    def create_scale_rxn_for_met(self, mod, base=10):
        scale_dic = self._check_scale(self.coef_in_rxns, bin=base)
        for scale_key, rxns in scale_dic.items():
            scale = int(scale_key)
            if scale == 0:
                continue
            add_scale_rxn_for_met(mod, self.met_id, rxns, c=base ** int(np.sign(scale)), scale_f=base ** scale)


def replace_met_completely(mod, met_id, scale_to):
    mets = [m for m in mod.metabolites if m.id[:len(met_id)] == met_id]

    for met in mets:
        new_met = cobra.Metabolite(id=met.id + f"_{scale_to}", name=met.name + f"_{scale_to}",
                                   compartment=met.compartment)
        new_met.elements = {k: v * scale_to for k, v in met.elements.items()}
        for r in met.reactions:
            oc = r.metabolites[met]
            r.add_metabolites({new_met: oc / scale_to})
            r.subtract_metabolites({met: oc})
    mod.remove_metabolites(mets)


def normalize_rxn(mod, rxn_id):
    rxn = mod.reactions.get_by_id(rxn_id)
    _, _, c = calc_rxn_unbalance_scores(rxn)
    new_met_dict = {m: coef / (10 ** c) for m, coef in rxn.metabolites.items()}
    rxn.subtract_metabolites(rxn.metabolites)
    rxn.add_metabolites(new_met_dict)
    rxn.bounds = [rxn.lower_bound * (10 ** c), rxn.upper_bound * (10 ** c)]


def total_rxn_unb_score(mod, rxn_ids):
    c = [0, 0, 0]
    for r_id in rxn_ids:
        c1, c2, c3 = calc_rxn_unbalance_scores(mod.reactions.get_by_id(r_id))
        c[0] += c1
        c[1] += c2
        c[2] += c3
    return c


def create_scaling_rxns(met_id, mod, rxns_in_mod=None, scale_to=100):
    rxn_name = f"scaling_rxn_for_{met_id}_{scale_to}"
    met = mod.metabolites.get_by_id(met_id)
    if rxns_in_mod is None:
        rxns_in_mod = [r.id for r in mod.reactions]

    if rxn_name in rxns_in_mod:
        # print("The reaction is already created")
        return mod.reactions.get_by_id(rxn_name), mod.metabolites.get_by_id(f"{met_id}_x_{scale_to}"), False

    rhs_met = cobra.Metabolite(f"{met_id}_x_{scale_to}", compartment=met.compartment)
    new_rxn = cobra.Reaction(rxn_name, name=rxn_name, lower_bound=-1e6, upper_bound=1e6)
    new_rxn.add_metabolites({rhs_met: 1 / scale_to, met: -1})
    # print(f"create a scaled met: {rhs_met.id} and a rxn: {new_rxn.id}")
    return new_rxn, rhs_met, True


def sub_met_to_scaled(rxn_id, met_id, mod, rxns_in_mod, scale_to):
    rxn = mod.reactions.get_by_id(rxn_id)
    new_rxn, rhs_met, is_new = create_scaling_rxns(met_id, mod, rxns_in_mod, scale_to)
    if is_new:
        mod.add_reactions([new_rxn])

    # sub met
    orig_c = rxn.metabolites[mod.metabolites.get_by_id(met_id)]
    rxn.subtract_metabolites({mod.metabolites.get_by_id(met_id): orig_c})
    rxn.add_metabolites({rhs_met: orig_c / scale_to})

    return new_rxn.id


def check_scales(mod, thres=1e3):
    bad_scales = {}

    for r in mod.reactions:
        max_r_c, min_r_c, max_l_c, min_l_c = 0, 1e6, 0, 1e6

        for m, c in r.metabolites.items():
            if c < 0:  # lhs
                max_l_c = max(abs(c), max_l_c)
                min_l_c = min(abs(c), min_l_c)
            else:
                max_r_c = max(abs(c), max_r_c)
                min_r_c = min(abs(c), min_r_c)

        if max_r_c / min_l_c > thres or max_l_c / min_r_c > thres:
            bad_scales[r.id] = {m.id: np.floor(np.log10(abs(c))) for m, c in r.metabolites.items()}
    return bad_scales


def check_rxn_scales(mod, threshold=1e4, base=10):
    # https://www.gurobi.com/documentation/9.5/refman/recommended_ranges_for_var.html
    for r in mod.reactions:
        normalize_rxn(mod, r.id)

    log_th = np.log10(threshold) / np.log10(base)
    for m in mod.metabolites:
        ms = MetScorer(m.id, mod)
        if ms.ub_scores >= log_th:
            ms.create_scale_rxn_for_met(mod)