import numpy as np
from pipeGEM import Model
from pipeGEM.data import GeneData


def test_MBA(ecoli_core, ecoli_core_data):
    # ecoli_core.solver = "gurobi"

    pmod = Model(model=ecoli_core, name_tag="ecoli")
    data_name = "sample_0"
    gene_data = GeneData(data=ecoli_core_data[data_name],
                         data_transform=lambda x: np.log2(x),
                         absent_expression=-np.inf)
    pmod.add_gene_data(data_name, gene_data)
    thres = gene_data.get_threshold("percentile", p=[50, 10])
    result = pmod.integrate_gene_data(data_name=data_name,
                                      integrator="MBA",
                                      predefined_threshold=thres,
                                      threshold_kws={},
                                      protected_rxns=["BIOMASS_Ecoli_core_w_GAM"])
    assert result is not None
    assert len(result.result_model.reactions) > 0
    assert isinstance(result.removed_rxn_ids, (list, set, tuple))
    # Kept reactions should be a subset of the original
    kept_ids = {r.id for r in result.result_model.reactions}
    original_ids = {r.id for r in ecoli_core.reactions}
    assert kept_ids <= original_ids
