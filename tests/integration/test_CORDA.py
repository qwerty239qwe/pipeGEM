import numpy as np
from pipeGEM import Model
from pipeGEM.data import GeneData


def test_CORDA(ecoli_core, ecoli_core_data):
    # ecoli_core.solver = "gurobi"

    pmod = Model(model=ecoli_core, name_tag="ecoli")
    data_name = "sample_0"
    gene_data = GeneData(data=ecoli_core_data[data_name],
                         data_transform=lambda x: np.log2(x),
                         absent_expression=-np.inf)
    pmod.add_gene_data(data_name, gene_data)
    thres = gene_data.get_threshold("percentile", p = [75, 25])
    print(gene_data.rxn_scores)
    print(thres.exp_th)

    result = pmod.integrate_gene_data(data_name=data_name,
                                      integrator="CORDA",
                                      predefined_threshold=thres,
                                      threshold_kws={},
                                      protected_rxns=["BIOMASS_Ecoli_core_w_GAM"])
    print("Kept: ", len(result.result_model.reactions))
    print("Removed: ", len(result.removed_rxn_ids))
    print("Conf scores: ", result.conf_scores)
    print("Efficacy: ", result.algo_efficacy)