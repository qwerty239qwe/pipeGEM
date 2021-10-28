from pipeGEM.core import batch, group, model


def test_get_model(ecoli_core):
    return model(model=ecoli_core, name="ecoli")


def test_get_group(ecoli_core):
    return group(named_models=[model(model=ecoli_core, name="ecoli"),
                               model(model=ecoli_core, name="ecoli2")])