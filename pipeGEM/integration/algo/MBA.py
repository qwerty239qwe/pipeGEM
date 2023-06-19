import numpy as np
from pipeGEM.analysis import timing, MBA_Analysis, consistency_testers, \
    NumInequalityStoppingCriteria, IsInSetStoppingCriteria
from pipeGEM.integration.utils import parse_predefined_threshold


@timing
def apply_MBA(model,
              data=None,
              predefined_threshold=None,
              threshold_kws: dict = None,
              protected_rxns=None,
              rxn_scaling_coefs: dict = None,
              medium_conf_rxn_ids=None,
              high_conf_rxn_ids=None,
              consistent_checking_method="FASTCC",
              tolerance=1e-8,
              epsilon=0.5,
              random_state=42):
    rxn_ids = [r.id for r in model.reactions]
    if data is not None:
        print(f"Using data-inferred threshold. Ignoring medium_conf_rxn_ids and high_conf_rxn_ids.")
        threshold_dic = parse_predefined_threshold(predefined_threshold,
                                                   gene_data=data.gene_data,
                                                   **threshold_kws)
        rxn_in_model = set([r.id for r in model.reactions])
        th_result, exp_th, non_exp_th = threshold_dic["th_result"], threshold_dic["exp_th"], threshold_dic["non_exp_th"]
        high_conf_rxn_ids = (set([r for r, c in data.rxn_scores.items() if c > exp_th]) | set(
            protected_rxns)) & rxn_in_model
        medium_conf_rxn_ids = (set([r for r, c in data.rxn_scores.items() if (c < exp_th) and (c > non_exp_th)]) - set(
            protected_rxns)) & rxn_in_model
    else:
        th_result= None
        print(f"Using defined medium and high conf rxns. Ignoring predefined_threshold.")
        assert all([r in rxn_ids for r in medium_conf_rxn_ids])
        assert all([r in rxn_ids for r in high_conf_rxn_ids])
    protected_rxns = protected_rxns if protected_rxns is not None else []
    high_conf_rxn_ids = list(set(high_conf_rxn_ids) | set(protected_rxns))
    medium_conf_rxn_ids = list(set(medium_conf_rxn_ids) - set(high_conf_rxn_ids))
    no_conf_set = np.array(list(set(rxn_ids) - set(medium_conf_rxn_ids) - set(high_conf_rxn_ids)))
    rng = np.random.default_rng(random_state)
    rng.shuffle(no_conf_set)
    removed_rxns = []
    kept_nc_rxns = []
    model = model.copy()

    for r in no_conf_set:
        if r in (removed_rxns + kept_nc_rxns):
            continue

        stop_crit = [IsInSetStoppingCriteria(ess_set=set(protected_rxns) | set(high_conf_rxn_ids)),
                     NumInequalityStoppingCriteria(var={"mc": list(set(medium_conf_rxn_ids) - set(removed_rxns)),
                                                        "nc": list(set(no_conf_set) - set(removed_rxns))},
                                                   cons_dict={"mc": -1,
                                                              "nc": epsilon})]
        with model:
            model.reactions.get_by_id(r).bounds = (0, 0)
            sol = model.optimize()
            if sol.status != "optimal":
                kept_nc_rxns.append(r)
                continue
            cons_tester = consistency_testers[consistent_checking_method](model)
            test_result = cons_tester.analyze(tol=tolerance,
                                              return_model=False,
                                              stopping_callback=stop_crit,
                                              rxn_scaling_coefs=rxn_scaling_coefs)
        excluded_HC = set(high_conf_rxn_ids) & set(test_result.removed_rxn_ids)
        excluded_MC = set(medium_conf_rxn_ids) & set(test_result.removed_rxn_ids)
        excluded_NC = set(test_result.removed_rxn_ids) - excluded_HC - excluded_MC

        if len(excluded_HC) == 0 and len(excluded_MC) < epsilon * len(excluded_NC):
            for removed_r in test_result.removed_rxn_ids:
                model.reactions.get_by_id(removed_r).bounds = (0, 0)
            removed_rxns.extend(test_result.removed_rxn_ids)
            print(f"Detect {len(test_result.removed_rxn_ids)} removable reactions in NC set.")
        else:
            kept_nc_rxns.append(r)

    model = model.remove_reactions(removed_rxns, remove_orphans=True)
    result = MBA_Analysis(log={"tolerance": tolerance,
                               "epsilon": epsilon,
                               "random_state": random_state})
    result.add_result(dict(result_model=model,
                           removed_rxn_ids=removed_rxns,
                           threshold_analysis=th_result))

    return result


