from pipeGEM.pipeline.algo import SwiftCC


def test_swiftcc(ecoli_core):
    s = SwiftCC()
    consist = s(ecoli_core)
    assert len(consist["model"].reactions) > 0, len(consist["model"].reactions) != len(ecoli_core.reactions)
    assert len(consist["model"].metabolites) > 0, len(consist["model"].metabolites) != len(ecoli_core.metabolites)
    assert len(consist["model"].genes) > 0, len(consist["model"].genes) != len(ecoli_core.genes)

