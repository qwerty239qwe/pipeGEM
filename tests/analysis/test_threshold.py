import pandas as pd

from pipeGEM.analysis import rFASTCORMICSThreshold, rFASTCORMICSThresholdAnalysis


def test_valid_input_data_returns_analysis_object():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    threshold_finder = rFASTCORMICSThreshold()
    result = threshold_finder.find_threshold(data=data, cut_off=0, return_heuristic=True)
    assert isinstance(result, rFASTCORMICSThresholdAnalysis)


def test_pd_series_input_data():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    threshold_finder = rFASTCORMICSThreshold()
    result = threshold_finder.find_threshold(data=data, cut_off=0, return_heuristic=True)
    assert isinstance(result, rFASTCORMICSThresholdAnalysis)


def test_empty_input_data_returns_empty_analysis_object():
    data = []
    threshold_finder = rFASTCORMICSThreshold()
    result = threshold_finder.find_threshold(data=data, cut_off=0, return_heuristic=True)
    assert isinstance(result, rFASTCORMICSThresholdAnalysis)
    assert len(result.result) == 0