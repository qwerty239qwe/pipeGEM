"""Real CPU inference with the bundled DLKcat weights; no prediction mocks."""
import numpy as np
import pandas as pd
import pytest
import cobra

pytest.importorskip("torch")
pytest.importorskip("rdkit")

from pipeGEM import Model
from pipeGEM.data import EnzymeData, MetaboliteData
from pipeGEM.data.data import ProteinAbundanceData
from pipeGEM.extensions.DLKcat import predict_Kcat


SEQUENCE = "MALWMRLLPLLALLALWGPDPAAA"


def test_real_dlkcat_prediction_preserves_labels():
    frame = pd.DataFrame([
        {"rxn": rxn, "genes": gene, "mets": met, "Smiles": "CCO", "Seq": SEQUENCE}
        for rxn, gene, met in [("R1", "g1", "a"), ("R2", "g2", "b")]
    ])
    predictions = predict_Kcat(frame, device="cpu")
    assert list(predictions.columns) == ["rxn", "gene", "met", "kcat"]
    assert predictions[["rxn", "gene", "met"]].values.tolist() == [
        ["R1", "g1", "a"], ["R2", "g2", "b"],
    ]
    assert np.isfinite(predictions["kcat"]).all()
    assert (predictions["kcat"] > 0).all()
    assert predictions["kcat"].iloc[0] == pytest.approx(predictions["kcat"].iloc[1])


@pytest.mark.parametrize("method", ["GECKOLight", "GECKOFull"])
def test_real_dlkcat_prediction_limits_gecko_flux(method):
    cobra_model = cobra.Model("dlkcat_test")
    a, b = (cobra.Metabolite(m, compartment="c") for m in ["a", "b"])
    uptake = cobra.Reaction("EX_a", lower_bound=0, upper_bound=1000)
    uptake.add_metabolites({a: 1})
    reaction = cobra.Reaction("R1", lower_bound=0, upper_bound=1000)
    reaction.add_metabolites({a: -1, b: 1})
    reaction.gene_reaction_rule = "g1"
    demand = cobra.Reaction("DM_b", lower_bound=0, upper_bound=1000)
    demand.add_metabolites({b: -1})
    cobra_model.add_reactions([uptake, reaction, demand])
    cobra_model.objective = demand

    model = Model(model=cobra_model)
    model.add_metabolite_data(MetaboliteData(pd.DataFrame({"SMILES": ["CCO"]}, index=["a"])))
    enzymes = EnzymeData(pd.DataFrame({
        "Protein": ["P1"], "Sequence": [SEQUENCE], "Kcat": [np.nan], "MW": [50_000.],
    }, index=["g1"]), prot_id_col="Protein")
    model.add_enzyme_data(enzymes, run_DLKcat=True, device="cpu")

    predicted = enzymes._enzyme_df["DLKcat"].iloc[0]
    assert np.isfinite(predicted) and predicted > 0
    assert pd.isna(enzymes._enzyme_df["Kcat"].iloc[0])
    assert enzymes.rxn_items()["R1"]["best_kcat"] == pytest.approx(predicted)

    abundance, sigma = 1e-6, 0.5
    model.add_protein_abundance_data("measured", ProteinAbundanceData(
        pd.DataFrame({"abundance": [abundance]}, index=["P1"]),
    ))
    result = model.integrate_enzyme_data("measured", method=method, sigma=sigma)
    solution = result.ec_model.optimize()
    assert solution.status == "optimal"
    assert solution.objective_value == pytest.approx(predicted * abundance * sigma * 3600)
    assert cobra_model.reactions.R1.bounds == (0, 1000)
