from pipeGEM.pipeline.algo import SwiftCC


def test_swiftcc(ecoli):
    s = SwiftCC()
    consist = s(ecoli)
    assert len(consist["model"].reactions) > 0, len(consist["model"].reactions) != len(ecoli.reactions)
    assert len(consist["model"].metabolites) > 0, len(consist["model"].metabolites) != len(ecoli.metabolites)
    assert len(consist["model"].genes) > 0, len(consist["model"].genes) != len(ecoli.genes)

