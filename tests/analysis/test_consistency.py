from pipeGEM.analysis import consistency_testers, \
    FASTCC, FVAConsistencyTester, FastCCAnalysis
from pipeGEM import Model
import cobra


def test_FASTCC_cobra_model(ecoli_core):
    fastcc = consistency_testers["FASTCC"](model=ecoli_core)
    assert isinstance(fastcc, FASTCC)
    fastcc_result = fastcc.analyze(tol=1e-6, return_model=True)
    assert isinstance(fastcc_result.consistent_model, cobra.Model)
    assert len(ecoli_core.reactions) > len(fastcc_result.consistent_model.reactions)


def test_FASTCC_pg_model(ecoli_core):
    fastcc = consistency_testers["FASTCC"](model=Model(model=ecoli_core, name_tag="ecoli"))
    assert isinstance(fastcc, FASTCC)
    fastcc_result = fastcc.analyze(tol=1e-6, return_model=True)
    assert isinstance(fastcc_result.consistent_model, Model)
    assert fastcc_result.consistent_model.name_tag == "consistent_ecoli"
    assert len(ecoli_core.reactions) > len(fastcc_result.consistent_model.reaction_ids)
    fastcc_result.save("./fastcc_result/")


def test_load_FASTCC_result():
    read_result = FastCCAnalysis.load("./fastcc_result/")
    assert isinstance(read_result.consistent_model, Model)


def test_FVA_pg_model(ecoli_core):
    fva = consistency_testers["FVA"](model=Model(model=ecoli_core, name_tag="ecoli"))
    assert isinstance(fva, FVAConsistencyTester)
    fva_result = fva.analyze(tol=1e-6, return_model=True)
    assert isinstance(fva_result.consistent_model, Model)
    assert fva_result.consistent_model.name_tag == "consistent_ecoli"
    assert len(ecoli_core.reactions) > len(fva_result.consistent_model.reaction_ids)


def test_method_return_the_same_result(ecoli_core):
    model = Model(model=ecoli_core, name_tag="ecoli")

    fva = consistency_testers["FVA"](model=model).analyze(tol=1e-6, return_model=True)
    fastcc = consistency_testers["FASTCC"](model=model).analyze(tol=1e-6, return_model=True)
    fva_r = set([r.id for r in fva.consistent_model.reactions])
    fastcc_r = set([r.id for r in fastcc.consistent_model.reactions])
    assert len(fva_r) == len(fastcc_r)
    assert len(fva_r - fastcc_r) == 0
