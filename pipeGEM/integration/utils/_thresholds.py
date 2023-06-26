from pipeGEM.analysis import rFASTCORMICSThreshold, PercentileThreshold, LocalThresholdAnalysis, \
    rFASTCORMICSThresholdAnalysis, PercentileThresholdAnalysis


def parse_predefined_threshold(predefined_threshold,
                               gene_data=None,
                               threshold_type_if_none: str = "rFASTCORMICS",
                               **kwargs):
    if predefined_threshold is None:
        if threshold_type_if_none == "rFASTCORMICS":
            th_finder = rFASTCORMICSThreshold()
            th_result = th_finder.find_threshold(gene_data,
                                                 **kwargs)
            non_exp_th = th_result.non_exp_th
            exp_th = th_result.exp_th
        elif threshold_type_if_none == "percentile":
            th_finder = PercentileThreshold()
            th_result = th_finder.find_threshold(gene_data,
                                                 **kwargs)
            non_exp_th = th_result.non_exp_th
            exp_th = th_result.exp_th

    else:
        th_result = predefined_threshold
        if isinstance(th_result, rFASTCORMICSThresholdAnalysis) or isinstance(th_result, PercentileThresholdAnalysis):
            non_exp_th = th_result.non_exp_th
            exp_th = th_result.exp_th
        elif isinstance(th_result, dict):
            exp_th, non_exp_th = th_result["exp_th"], th_result["non_exp_th"]
        else:
            non_exp_th = th_result["non_exp_th"].exp_th if hasattr(th_result["non_exp_th"], "exp_th") else th_result["non_exp_th"]
            exp_th = th_result["exp_th"].exp_th if hasattr(th_result["exp_th"], "exp_th") else th_result["exp_th"]
    return {"th_result": th_result,
            "exp_th": exp_th,
            "non_exp_th": non_exp_th}