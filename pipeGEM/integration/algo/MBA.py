import numpy as np
from pipeGEM.integration.algo.fastcore import fastcc
from pipeGEM.analysis import timing, MBA_Analysis


@timing
def apply_MBA(model,
              medium_conf_rxn_ids,
              high_conf_rxn_ids,
              flux_threshold=1e-8,
              epsilon=0.5,
              random_state=42):
    rxn_ids = [r.id for r in model.reactions]
    assert all([r in rxn_ids for r in medium_conf_rxn_ids])
    assert all([r in rxn_ids for r in high_conf_rxn_ids])

    no_conf_set = np.array(list(set(rxn_ids) - set(medium_conf_rxn_ids) - set(high_conf_rxn_ids)))
    rng = np.random.default_rng(random_state)
    rng.shuffle(no_conf_set)
    removed_rxns = []
    kept_nc_rxns = []
    model = model.copy()

    for r in no_conf_set:
        if r in (removed_rxns + kept_nc_rxns):
            continue
        with model:
            model.reactions.get_by_id(r).bounds = (0, 0)
            sol = model.optimize()
            if sol.status != "optimal":
                kept_nc_rxns.append(r)
                continue

            fastcc_output = fastcc(model, epsilon=flux_threshold,
                                   return_model=False,
                                   return_removed_rxn_ids=True,
                                   return_rxn_ids=False)
        excluded_HC = set(high_conf_rxn_ids) & set(fastcc_output["removed_rxn_ids"])
        excluded_MC = set(medium_conf_rxn_ids) & set(fastcc_output["removed_rxn_ids"])
        excluded_NC = set(medium_conf_rxn_ids) & set(fastcc_output["removed_rxn_ids"])

        if len(excluded_HC) == 0 and len(excluded_MC) < epsilon * len(excluded_NC):
            for removed_r in fastcc_output["removed_rxn_ids"]:
                model.reactions.get_by_id(removed_r).bounds = (0, 0)
            removed_rxns.extend(fastcc_output["removed_rxn_ids"])
            print(f"Detect {fastcc_output['removed_rxn_ids']} removable reactions in NC set.")
        else:
            kept_nc_rxns.append(r)

    model = model.remove_reactions(removed_rxns, remove_orphans=True)
    result = MBA_Analysis(log={"flux_threshold": flux_threshold,
                               "epsilon": epsilon,
                               "random_state": random_state})
    result.add_result(model = model, removed_rxns=removed_rxns, threshold_analysis=None)

    return result


