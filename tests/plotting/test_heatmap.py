import pandas as pd
from pipeGEM.plotting.heatmap import _parse_one_axis_colors


def test_parse_one_axis_colors():
    dummy_groups = pd.DataFrame({
        "treatment": ["a", "b", "a", "b", "c"],
        "cell_type": ["1", "2", "1", "3", "4"]
    }, index=["sample1", "sample2", "sample3", "sample4", "sample5"])

    print(_parse_one_axis_colors(dummy_groups))