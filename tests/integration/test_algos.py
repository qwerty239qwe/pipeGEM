import numpy as np

from pipeGEM import Model
from pipeGEM.analysis import ProblemAnalyzer
from pipeGEM.integration.algo.swiftcore import swiftcc, swiftCore
from pipeGEM.integration.algo.swiftcore import CoreProblem


def test_swiftcc(ecoli_core):
    consist = swiftcc(Model(ecoli_core, "ecoli"))
    print(len(ecoli_core.reactions))
    print(consist)


def test_swiftProblem(ecoli_core):
    ecoli = Model(ecoli_core, "ecoli")
    weights = np.ones(shape=(len(ecoli.reactions),))
    #core_index = np.random.choice(len(ecoli.reactions), 50, replace=False)
    core_index = [i for i, r in enumerate(ecoli_core.reactions) if r.id == "BIOMASS_Ecoli_core_w_GAM"]
    is_core = np.array([(i in core_index) for i in range(len(ecoli.reactions))])
    m, n = len(ecoli.metabolites), len(ecoli.reactions)
    blocked = np.array([False for _ in range(n)])

    # assert sum(ecoli.optimize().to_frame()["fluxes"].values) != 0, sum(ecoli.optimize().to_frame()["fluxes"].values)
    problem = CoreProblem(model=ecoli,
                          blocked=blocked,
                          weights=weights,
                          core_index=is_core,
                          do_flip=True,
                          do_reduction=False)
    rxn_ids = np.array([rxn.id for rxn in ecoli.reactions if not rxn.reversibility])
    core_model = ProblemAnalyzer(problem)
    flux = core_model.get_fluxes("min")
    # print(flux)
    print(flux.iloc[:n, :][flux.iloc[:n, :]["fluxes"] != 0])


def test_q(ecoli):
    return ecoli.optimize().to_frame()["fluxes"].values


def test_swiftCore(ecoli_core):
    # consis = swiftcc(Model(ecoli_core, "ecoli"), return_model=True)
    # core_index = np.random.choice(len(consis.reactions), 30, replace=False)
    core_index = [i for i, r in enumerate(ecoli_core.reactions) if r.id == "BIOMASS_Ecoli_core_w_GAM"]
    core_rxns = np.array([rxn.id for rxn in ecoli_core.reactions])[core_index]
    print(core_rxns)
    result = swiftCore(Model(ecoli_core, "ecoli"), core_index=core_index)
    assert result.optimize()
    assert len(set(core_rxns) - set([r.id for r in result.reactions])) == 0
    print(len(result.reactions))
    print(result.optimize())
    assert len(set([r.id for r in result.reactions])) < len(ecoli_core.reactions)