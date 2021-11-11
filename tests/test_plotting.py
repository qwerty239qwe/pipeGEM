import pandas as pd

import pipeGEM as pg
from pipeGEM.plotting._prep import prep_fva_plotting_data
from pipeGEM.plotting._flux import plot_fva


def test_prep_FVA():
    min_df = pd.DataFrame({"Reaction_1": [1, 2, 3, 4, 5],
                           "Reaction_2": [3, 4, 5, 6, 7]},
                          index=[f"model_{i}" for i in range(5)])
    max_df = pd.DataFrame({"Reaction_1": [3, 4, 6, 8, 10],
                           "Reaction_2": [10, 20, 30, 40, 50]},
                          index=[f"model_{i}" for i in range(5)])
    model_info = pd.DataFrame({"model_name": ["m1", "m2", "m3", "m4", "m5"],
                               "group": ["g1", "g1", "g2", "g2", "g3"]})

    result = prep_fva_plotting_data(min_df=min_df, max_df=max_df, model_info=model_info["group"])
    assert result.shape == (5*2*5, 3), result


def test_plot_FVA():
    min_df = pd.DataFrame({"Reaction_1": [1, 2, 3, 4, 5],
                           "Reaction_2": [3, 4, 5, 6, 7],
                           "model_name": ["m1", "m2", "m3", "m4", "m5"],
                           "group": ["g1", "g1", "g2", "g2", "g3"]
                           },
                          index=[f"model_{i}" for i in range(5)])
    max_df = pd.DataFrame({"Reaction_1": [3, 4, 6, 8, 10],
                           "Reaction_2": [10, 20, 30, 40, 50],
                           "model_name": ["m1", "m2", "m3", "m4", "m5"],
                           "group": ["g1", "g1", "g2", "g2", "g3"]
                           },
                          index=[f"model_{i}" for i in range(5)])

    plot_fva(min_df, max_df, rxn_ids=["Reaction_1", "Reaction_2"], group_layer="group")


def test_plot_components(ecoli_core, ecoli):
    g = pg.Group({"ecoli": ecoli, "ecoli_core": ecoli_core})
    g.plot_components(group_order=["ecoli", "ecoli_core"])