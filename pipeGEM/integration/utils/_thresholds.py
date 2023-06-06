from pipeGEM.analysis import rFASTCORMICSThreshold, rFASTCORMICSThresholdAnalysis


def parse_predefined_threshold(predefined_threshold,
                               gene_data=None,
                               use_heuristic_th=False):
    if predefined_threshold is None:
        th_finder = rFASTCORMICSThreshold()
        th_result = th_finder.find_threshold(gene_data, return_heuristic=use_heuristic_th)
        non_exp_th = th_result.non_exp_th
        exp_th = th_result.exp_th
    else:
        th_result = predefined_threshold
        if isinstance(th_result, rFASTCORMICSThresholdAnalysis):
            non_exp_th = th_result.non_exp_th
            exp_th = th_result.exp_th
        else:
            non_exp_th = th_result["non_exp_th"].exp_th if hasattr(th_result["non_exp_th"], "exp_th") else th_result["non_exp_th"]
            exp_th = th_result["exp_th"].exp_th if hasattr(th_result["exp_th"], "exp_th") else th_result["exp_th"]
    return {"th_result": th_result, "exp_th": exp_th, "non_exp_th": non_exp_th}