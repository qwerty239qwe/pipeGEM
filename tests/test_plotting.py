import pandas as pd

from pipeGEM.plotting._prep import prep_fva_plotting_data
from pipeGEM.plotting._flux import plot_fva


def test_prep_FVA():
    min_df = pd.DataFrame({"Reaction_1": [1, 2, 3, 4, 5],
                           "Reaction_2": [3, 4, 5, 6, 7]},
                          index=[f"model_{i}" for i in range(5)])
    max_df = pd.DataFrame({"Reaction_1": [3, 4, 6, 8, 10],
                           "Reaction_2": [10, 20, 30, 40, 50]},
                          index=[f"model_{i}" for i in range(5)])

    result = prep_fva_plotting_data(min_df=min_df, max_df=max_df)
    assert result.shape == (5*2*5, 3), result


def test_plot_FVA():
    min_df = pd.DataFrame({"Reaction_1": [1, 2, 3, 4, 5],
                           "Reaction_2": [3, 4, 5, 6, 7]},
                          index=[f"model_{i}" for i in range(5)])
    max_df = pd.DataFrame({"Reaction_1": [3, 4, 6, 8, 10],
                           "Reaction_2": [10, 20, 30, 40, 50]},
                          index=[f"model_{i}" for i in range(5)])

    plot_fva(min_df, max_df, rxn_ids=["Reaction_1", "Reaction_2"])