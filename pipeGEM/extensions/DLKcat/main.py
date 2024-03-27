from pipeGEM.extensions.DLKcat.model import *
from pipeGEM.extensions.DLKcat.utils import *
from pipeGEM.extensions.DLKcat.preprocess import *
from pipeGEM.extensions.DLKcat.data import *
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import torch
from pathlib import Path


DEFAULT_PRECOMP = Path(__file__).parent / "precomputed_obj"


def _prepare_preditor_and_data(df, device="cpu", precom_obj_dir=DEFAULT_PRECOMP):
    fingerprint_dict = load_pickle(Path(precom_obj_dir) / 'fingerprint_dict.pickle')
    atom_dict = load_pickle(Path(precom_obj_dir) / 'atom_dict.pickle')
    bond_dict = load_pickle(Path(precom_obj_dir) / 'bond_dict.pickle')
    word_dict = load_pickle(Path(precom_obj_dir) / 'sequence_dict.pickle')
    edge_dict = load_pickle(Path(precom_obj_dir) / 'edge_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    Kcat_model = KcatPrediction(device, n_fingerprint, n_word)
    Kcat_model.load_state_dict(torch.load(Path(precom_obj_dir) / 'model_state'))
    predictor = Predictor(Kcat_model)
    dataset = SeqSmileDataset(fingerprint_dict=fingerprint_dict,
                              atom_dict=atom_dict,
                              bond_dict=bond_dict,
                              word_dict=word_dict,
                              edge_dict=edge_dict,
                              df=df)
    return predictor, dataset


def predict_Kcat(df, device="cpu") -> pd.DataFrame:
    predictor, dataset = _prepare_preditor_and_data(df=df, device=device)
    pred_dl = DataLoader(dataset, batch_size=1)

    results = []
    for batch in tqdm(pred_dl):
        labels, inputs = batch["label"], batch["data"]
        if not isinstance(inputs[0], torch.Tensor) or (inputs[0].shape == inputs[1].shape == inputs[2].shape):
            results.append({**dict(zip(["rxn", "gene", "met"], labels)), **dict(kcat=None)})
            continue

        inputs = [inp.squeeze() for inp in inputs]
        try:
            prediction = predictor.predict(inputs)
            Kcat_log_value = prediction.item()
            Kcat_value = 2 ** Kcat_log_value
        except KeyError:
            print(labels, " has keyerr")
            Kcat_value = None
        results.append({**dict(zip(["rxn", "gene", "met"], labels)), **dict(kcat=Kcat_value)})

    result_df = pd.DataFrame(results)
    result_df["rxn"] = result_df["rxn"].apply(lambda x: x[0])
    result_df["gene"] = result_df["gene"].apply(lambda x: x[0])
    result_df["met"] = result_df["met"].apply(lambda x: x[0])
    return result_df