import pytest
import matplotlib


@pytest.fixture(scope="module")
def group_pfba_result(group):
    yield group.do_flux_analysis(method="pFBA", solver="glpk")


@pytest.fixture(scope="module")
def group_pfba_result_gb_gp(group):
    yield group.do_flux_analysis(method="pFBA", group_by="group_name", solver="glpk")


def test_plot_three_rxn_fluxes(group, group_pfba_result):
    rxns = group["m111"].reaction_ids[:3]
    result = group_pfba_result.plot(rxn_ids=rxns, group_by="model", aspect=1.5, kind="bar")
    # Should not error; result may be a figure/axes or None
    if result is not None:
        assert isinstance(result, (matplotlib.figure.Figure, matplotlib.axes.Axes)) or hasattr(result, "fig")


def test_plot_three_rxn_fluxes_gb_gp(group, group_pfba_result_gb_gp):
    rxns = group["m111"].reaction_ids[:3]
    result = group_pfba_result_gb_gp.plot(rxn_ids=rxns, group_by="group_name", aspect=1.5, kind="bar")
    if result is not None:
        assert isinstance(result, (matplotlib.figure.Figure, matplotlib.axes.Axes)) or hasattr(result, "fig")
