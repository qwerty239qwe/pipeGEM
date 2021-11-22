import numpy as np

from pipeGEM import Model
from pipeGEM.integration.algo.swiftcore import swiftcc, swiftCore
from pipeGEM.integration.algo.swiftcore import CoreProblem


def test_swiftcc(ecoli):
    print(len(ecoli.reactions), sum(swiftcc(Model(ecoli, "ecoli"))))


def test_swiftProblem(ecoli):
    ecoli = Model(ecoli, "ecoli")
    weights = np.ones(shape=(len(ecoli.reactions),))
    consistent = swiftcc(ecoli)
    core_index = np.random.choice(len(ecoli.reactions), 500, replace=False)
    is_core = np.array([(i in core_index) for i in range(len(ecoli.reactions))])
    __w = len(is_core)
    print(len(is_core))
    print(sum(consistent))
    m, n = len(ecoli.metabolites), len(ecoli.reactions)
    assert sum(ecoli.optimize().to_frame()["fluxes"].values) != 0, sum(ecoli.optimize().to_frame()["fluxes"].values)
    problem = CoreProblem(model=ecoli,
                          consistent=consistent,
                          weights=weights,
                          core_index=is_core,
                          do_flip=False,
                          do_reduction=False)
    core_model = problem.to_model("core", direction="max")
    print(problem.objs[problem.objs != 0])
    flux = core_model.get_problem_fluxes()
    print(flux[flux["fluxes"] != 0])


def test_q(ecoli):
    return ecoli.optimize().to_frame()["fluxes"].values


def test_swiftCore(ecoli):
    core_index = np.random.choice(len(ecoli.reactions), 100, replace=False)
    result = swiftCore(Model(ecoli, "ecoli"), core_index=core_index)
    assert result.optimize()