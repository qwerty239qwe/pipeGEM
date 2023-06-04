import pandas as pd
import tomlkit
import numpy as np
from pathlib import Path
import scanpy as sc


dense_data_loader = {"csv": pd.read_csv,
                     "tsv": pd.read_csv,
                     "txt": pd.read_csv,
                     "xlsx": pd.read_excel,
                     "json": pd.read_json,
                     "parquet": pd.read_parquet,
                     "fwf": pd.read_fwf}
sparse_data_loader = {"h5ad": sc.read_h5ad,
                      "txt": sc.read_text}


def load_data(file_path,
              data_type='infer',
              parser="csv",
              **kwargs):
    file_path = Path(file_path)

    if data_type == 'infer':
        if file_path.suffix[1:] in dense_data_loader:
            return dense_data_loader[file_path.suffix[1:]](file_path, **kwargs)
        elif file_path.suffix[1:] in sparse_data_loader:
            return sparse_data_loader[file_path.suffix[1:]](file_path, **kwargs)
        else:
            raise ValueError(f"cannot infer file with suffix: {file_path.suffix},"
                       f"please manually select data_type and parser")
    elif data_type == "dense":
        return dense_data_loader[parser](file_path, **kwargs)
    elif data_type == "sparse":
        return sparse_data_loader[parser](file_path, **kwargs)
    raise ValueError(r"Please select data_type from 'infer', 'dense', and 'sparse'")