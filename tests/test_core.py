from pipeGEM.core import batch, group, model


def test_get_model(ecoli_core):
    return model(model=ecoli_core, name="ecoli")


def test_get_group(ecoli_core):
    return group(named_models=[model(model=ecoli_core, name="ecoli"),
                               model(model=ecoli_core, name="ecoli2")])


def test_get_group_2(ecoli_core):
    return group(named_models={"ecoli_1": ecoli_core,
                               "ecoli_2": ecoli_core})