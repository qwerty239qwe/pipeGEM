import numpy as np
from tqdm import tqdm
from pipeGEM.analysis import timing, MBA_Analysis, consistency_testers, \
    NumInequalityStoppingCriteria, IsInSetStoppingCriteria, measure_efficacy
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
              consistent_checking_method: str = "FASTCC",
              tolerance: float = 1e-8,
              epsilon: float = 0.33,
              random_state: int = 42):
    """Apply the Model Building Algorithm (MBA) to generate a context-specific model.

    MBA iteratively removes reactions with no confidence score ('no-confidence' set)
    based on consistency checks, while preserving high-confidence reactions and
    minimizing the removal of medium-confidence reactions.

    Parameters
    ----------
    model : cobra.Model
        The input genome-scale metabolic model.
    data : object, optional
        An object containing gene expression data (`data.gene_data`) and
        reaction scores (`data.rxn_scores`). If provided, `medium_conf_rxn_ids`
        and `high_conf_rxn_ids` are derived from this data using thresholds.
        Defaults to None.
    predefined_threshold : dict or analysis_types, optional
        Strategy or dictionary defining thresholds (`exp_th`, `non_exp_th`) to
        classify reactions based on scores when `data` is provided. See
        `pipeGEM.integration.utils.parse_predefined_threshold`. Defaults to None.
    threshold_kws : dict, optional
        Additional keyword arguments for the thresholding function when `data`
        is provided. Defaults to None.
    protected_rxns : list[str], optional
        A list of reaction IDs that should always be treated as high-confidence
        and never removed. Defaults to None.
    rxn_scaling_coefs : dict[str, float], optional
        Dictionary mapping reaction IDs to scaling coefficients, used to adjust
        consistency check tolerance. Defaults to None.
    medium_conf_rxn_ids : list[str], optional
        List of reaction IDs considered medium confidence. Used only if `data`
        is None. Defaults to None.
    high_conf_rxn_ids : list[str], optional
        List of reaction IDs considered high confidence. Used only if `data`
        is None. Defaults to None.
    consistent_checking_method : str, optional
        Method used for consistency checks (e.g., 'FASTCC'). Defaults to "FASTCC".
    tolerance : float, optional
        Tolerance used for consistency checks. Defaults to 1e-8.
    epsilon : float, optional
        Weighting factor used in the consistency check stopping criteria.
        Represents the maximum allowed ratio of removed medium-confidence
        reactions to removed no-confidence reactions during the removal check
        of a no-confidence reaction. Defaults to 0.33.
    random_state : int, optional
        Seed for the random number generator used to shuffle the order of
        no-confidence reactions being tested for removal. Defaults to 42.

    Returns
    -------
    MBA_Analysis
        An object containing the results:
        - result_model (cobra.Model): The final context-specific model.
        - removed_rxn_ids (np.ndarray): IDs of removed reactions.
        - threshold_analysis (ThresholdAnalysis or None): Details of thresholding
          used if `data` was provided.
        - algo_efficacy (float): Efficacy score comparing the final model
          against the initial high/no-confidence sets.

    Raises
    ------
    AssertionError
        If `data` is None and either `medium_conf_rxn_ids` or
        `high_conf_rxn_ids` contain IDs not present in the model.

    Notes
    -----
    Based on the algorithm described in: Jerby, L., Shlomi, T., & Ruppin, E. (2010).
    Computational reconstruction of tissue-specific metabolic models: application
    to human tissues. Molecular systems biology, 6(1), 401.
    """
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
        th_result = None
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

    for r in tqdm(no_conf_set):
        if r in (removed_rxns + kept_nc_rxns):
            continue

        stop_crit = [IsInSetStoppingCriteria(ess_set=set(high_conf_rxn_ids)),
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
        if "stopped" in test_result.log and test_result.log["stopped"]:
            continue

        excluded_HC = set(high_conf_rxn_ids) & set(test_result.removed_rxn_ids)
        excluded_MC = set(medium_conf_rxn_ids) & set(test_result.removed_rxn_ids)
        excluded_NC = set(test_result.removed_rxn_ids) - excluded_HC - excluded_MC
        print(excluded_HC, excluded_MC, excluded_NC)

        if len(excluded_HC) == 0 and len(excluded_MC) < epsilon * len(excluded_NC):
            for removed_r in test_result.removed_rxn_ids:
                model.reactions.get_by_id(removed_r).bounds = (0, 0)
            removed_rxns.extend(test_result.removed_rxn_ids)
            print(f"Detect {len(test_result.removed_rxn_ids)} removable reactions in NC set.")
        else:
            kept_nc_rxns.append(r)
    removed_rxns = list(set(removed_rxns))
    model.remove_reactions(removed_rxns, remove_orphans=True)
    eff_score = measure_efficacy(kept_rxn_ids=[r.id for r in model.reactions],
                                 removed_rxn_ids=removed_rxns,
                                 core_rxn_ids=high_conf_rxn_ids,
                                 non_core_rxn_ids=no_conf_set)
    result = MBA_Analysis(log={"tolerance": tolerance,
                               "epsilon": epsilon,
                               "random_state": random_state})
    result.add_result(dict(result_model=model,
                           removed_rxn_ids=np.array(removed_rxns),
                           threshold_analysis=th_result,
                           algo_efficacy=eff_score))

    return result
