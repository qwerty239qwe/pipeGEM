import pytest


@pytest.fixture(scope="module")
def group_pfba_result(group):
    yield group.do_flux_analysis(method="pFBA", solver="glpk")


@pytest.fixture(scope="module")
def group_pfba_result_gb_gp(group):
    yield group.do_flux_analysis(method="pFBA", group_by="group_name", solver="glpk")


def test_plot_three_rxn_fluxes(group, group_pfba_result):
    rxns = group["m111"].reaction_ids[:3]
    group_pfba_result.plot(rxn_ids=rxns, group_by="model", aspect=1.5, kind="bar")


def test_plot_three_rxn_fluxes_gb_gp(group, group_pfba_result_gb_gp):
    rxns = group["m111"].reaction_ids[:3]
    group_pfba_result_gb_gp.plot(rxn_ids=rxns, group_by="group_name", aspect=1.5, kind="bar")