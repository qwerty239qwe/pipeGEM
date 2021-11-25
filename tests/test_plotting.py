import pandas as pd

import pipeGEM as pg
from pipeGEM.plotting._prep import prep_fva_plotting_data
from pipeGEM.plotting._flux import plot_fva
from pipeGEM.utils import random_perturb


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


def test_plot_fba(ecoli_core):
    g = pg.Group({"ecoli": ecoli_core, "ecoli2": ecoli_core.copy()})

    g.do_analysis(method="FBA", constr="default")
    g.plot_flux(method="FBA", constr="default", rxn_ids=['PFK', 'PGI', 'PGK', 'EX_lac__D_e', 'PGL'],
                verbosity=1)


def test_plot_sampling(ecoli_core):
    g = pg.Group({"ecoli": ecoli_core, "ecoli2": ecoli_core.copy()})

    g.do_analysis(method="sampling", constr="default", n=50)
    g.plot_flux(method="sampling", constr="default", rxn_ids=['PFK', 'PGI', 'PGK'])


def test_plot_model_heatmap(ecoli_core):
    g = pg.Group({"ecoli": ecoli_core, "ecoli2": ecoli_core.copy(), "ecoli3": ecoli_core.copy()})
    g.plot_model_heatmap()


def test_plot_model_emb(ecoli_core):
    ecoli_2 = ecoli_core.copy()
    ecoli_2.reactions.get_by_id('PFK').knock_out()
    ecoli_3 = ecoli_2.copy()
    ecoli_3.reactions.get_by_id("PGM").knock_out()

    g = pg.Group({"ecoli": ecoli_core, "ecoli2": ecoli_2, "ecoli3": ecoli_3})
    g.do_analysis(method="sampling", constr="default", n=10)
    g.plot_flux_emb(method="sampling", constr="default", title="PCA")
    g.plot_flux_emb(method="sampling", constr="default", dr_method="UMAP", title="UMAP")


def test_plot_flux_heatmap(ecoli_core):
    ecoli_2 = ecoli_core.copy()
    ecoli_2 = random_perturb(ecoli_2, structure_ratio=0.98, constr_ratio=0.9)
    ecoli_3 = ecoli_2.copy()
    ecoli_3 = random_perturb(ecoli_3, structure_ratio=1, constr_ratio=0.9)

    g = pg.Group({"ecoli": {"mod1_1": ecoli_core, "mod2_1": random_perturb(ecoli_core,
                                                                           in_place=False,
                                                                           structure_ratio=0.98,
                                                                           constr_ratio=0.9)},
                  "ecoli2": {"mod1_2": ecoli_2, "mod2_2": random_perturb(ecoli_2, in_place=False,
                                                                         structure_ratio=1,
                                                                         constr_ratio=0.4)},
                  "ecoli3": {"mod1_3": ecoli_3, "mod2_3": random_perturb(ecoli_3, in_place=False,
                                                                         structure_ratio=1,
                                                                         constr_ratio=0.55)} })
    g.do_analysis(method="FBA", constr="default")
    g.plot_flux_heatmap(method="FBA", constr="default", get_model_level=True)