import numpy as np

from pipeGEM import Model
from pipeGEM.integration.algo.swiftcore import swiftcc, swiftCore
from pipeGEM.integration.algo.swiftcore import CoreProblem


def test_swiftcc(ecoli):
    print(len(ecoli.reactions), sum(swiftcc(Model(ecoli, "ecoli"))))


def test_swiftProblem(ecoli_core):
    ecoli = Model(ecoli_core, "ecoli")
    weights = np.ones(shape=(len(ecoli.reactions),))
    core_index = np.random.choice(len(ecoli.reactions), 50, replace=False)
    is_core = np.array([(i in core_index) for i in range(len(ecoli.reactions))])
    m, n = len(ecoli.metabolites), len(ecoli.reactions)
    blocked = np.array([False for _ in range(n)])

    assert sum(ecoli.optimize().to_frame()["fluxes"].values) != 0, sum(ecoli.optimize().to_frame()["fluxes"].values)
    problem = CoreProblem(model=ecoli,
                          blocked=blocked,
                          weights=weights,
                          core_index=is_core,
                          do_flip=True,
                          do_reduction=False)
    rxn_ids = np.array([rxn.id for rxn in ecoli.reactions if not rxn.reversibility])
    core_model = problem.to_model("core", direction="min")
    flux = core_model.get_problem_fluxes("min")
    print(flux.iloc[:n, :][flux.iloc[:n, :]["fluxes"] != 0])
    print(sorted(rxn_ids))


def test_q(ecoli):
    return ecoli.optimize().to_frame()["fluxes"].values


def test_swiftCore(ecoli_core):
    consis = swiftcc(Model(ecoli_core, "ecoli"), return_model=True)
    core_index = np.random.choice(len(consis.reactions), 10, replace=False)
    core_rxns = np.array([rxn.id for rxn in consis.reactions])[core_index]
    print(core_rxns)
    result = swiftCore(Model(consis, "ecoli"), core_index=core_index)
    assert result.optimize()
    assert len(set(core_rxns) - set([r.id for r in result.reactions])) == 0
    print(len(result.reactions))
    print(result.optimize())
    assert len(set([r.id for r in result.reactions])) < len(consis.reactions)