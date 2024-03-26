from torch.utils.data import Dataset
from .preprocess import *
from rdkit import Chem
import torch
import pandas as pd


class SeqSmileDataset(Dataset):
    def __init__(self, 
                 fingerprint_dict,
                 atom_dict,
                 bond_dict,
                 word_dict,
                 edge_dict,
                 df=None, 
                 data_lis=None, 
                 radius=2, 
                 ngram=3,
                 ):
        super().__init__()
        self.df = df.copy() if df is not None else pd.DataFrame(data_lis)
        assert len({"Smiles", "Seq"} - set(self.df.columns)) == 0, self.df.columns
        self.radius = radius
        self.ngram = ngram
        self.fingerprint_dict = fingerprint_dict
        self.atom_dict = atom_dict
        self.bond_dict = bond_dict
        self.word_dict = word_dict
        self.edge_dict = edge_dict
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, ind):
        data = self.df.iloc[ind, :]
        if pd.isna(data["Smiles"]) or "." in data["Smiles"] or data["Smiles"] == "":
            return {"label": [data["rxn"], data["genes"], data["mets"]],
                    "data": [-1, -1, -1]}
        try:
            mol = Chem.AddHs(Chem.MolFromSmiles(data["Smiles"]))
            atoms = create_atoms(mol,atom_dict=self.atom_dict)
            i_jbond_dict = create_ijbonddict(mol, self.bond_dict)
            fingerprints = torch.LongTensor(extract_fingerprints(atoms, i_jbond_dict, self.radius, self.fingerprint_dict, self.edge_dict))
            adjacency = torch.FloatTensor(create_adjacency(mol))
        except KeyError as e:
            print(e)
            print(f"prob smiles: {data['Smiles']}")
            return {"label": [data["rxn"], data["genes"], data["mets"]],
                    "data": [-1, -1, -1]}
        words = torch.LongTensor(split_sequence(data["Seq"], self.ngram, self.word_dict))
        return {"label": [data["rxn"], data["genes"], data["mets"]],
                "data": [fingerprints, adjacency, words]}
        