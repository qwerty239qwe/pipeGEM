import zipfile
from pathlib import Path
from io import BytesIO

import requests


__all__ = ["fetch_HPA_data"]


def fetch_and_extract(url, saved_path):
    resp = requests.get(url, stream=True)
    content = resp.content
    zf = zipfile.ZipFile(BytesIO(content))
    with zf as f:
        f.extractall(saved_path)


def fetch_HPA_data(options,
                   saved_path: Path = Path(__file__).parent.parent.parent / Path("assets/gene_data/HPA")):
    data_options = ["normal_tissue",
                    "pathology",
                    "subcellular_location",
                    "rna_tissue_consensus",
                    "rna_tissue_hpa",
                    "rna_tissue_gtex",
                    "rna_tissue_fantom",
                    "rna_single_cell_type",
                    "rna_single_cell_type_tissue",
                    "rna_brain_gtex",
                    "rna_brain_fantom",
                    "rna_pig_brain_hpa",
                    "rna_pig_brain_sample_hpa",
                    "rna_mouse_brain_hpa",
                    "rna_mouse_brain_sample_hpa",
                    "rna_mouse_brain_allen",
                    "rna_blood_cell",
                    "rna_blood_cell_sample",
                    "rna_blood_cell_sample_tpm_m",
                    "rna_blood_cell_monaco",
                    "rna_blood_cell_schmiedel",
                    "rna_celline",
                    "rna_cancer_sample",
                    "transcript_rna_tissue",
                    "transcript_rna_celline",
                    "transcript_rna_pigbrain",
                    "transcript_rna_mousebrain"]
    for opt in options:
        if opt in data_options:
            fetch_and_extract(url=f"https://www.proteinatlas.org/download/{opt}.tsv.zip", saved_path=saved_path)
            print("Data is downloaded and saved in ", saved_path)