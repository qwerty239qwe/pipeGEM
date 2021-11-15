from pipeGEM import Model
from pipeGEM.integration.algo.swiftcore import swiftcc


def test_swiftcc(ecoli):
    print(len(ecoli.reactions), sum(swiftcc(Model(ecoli, "ecoli"))))