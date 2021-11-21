import numpy as np

from pipeGEM import Model
from pipeGEM.integration.algo.swiftcore import swiftcc, swiftCore


def test_swiftcc(ecoli):
    print(len(ecoli.reactions), sum(swiftcc(Model(ecoli, "ecoli"))))


def test_swiftCore(ecoli):
    core_index = np.random.choice(len(ecoli.reactions), 100, replace=False)
    result = swiftCore(Model(ecoli, "ecoli"), core_index=core_index)
    assert result.optimize()