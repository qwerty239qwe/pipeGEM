from pathlib import Path
from typing import Dict, List, Union
import requests

import pandas as pd
from biodbs.HPA import HPAdb


def fetch_HPA_data(data_name,
                   data_path=Path(__file__).parent.parent.parent / Path("external_data/HPA")) -> dict:
    if isinstance(data_path, str):
        data_path = Path(data_path)

    data_path.mkdir(parents=True, exist_ok=True)
    if not (data_path / Path(data_name)).with_suffix(".tsv").exists():
        print("fetching data...")
        hpa = HPAdb()
        hpa.download_HPA_data(options=[data_name], saved_path=data_path)
    else:
        print("The dataframe is already exist.")
    return {"data_path": (data_path / Path(data_name)).with_suffix(".tsv")}


def load_HPA_data(data_path,
                  gene_col: str,
                  df_query_kw: Dict[str, Union[str, List[str]]] = None,
                  ):
    """
    Load and add translate column HPA data from a directory
    Returns
    -------
    """
    if df_query_kw is None:
        df_query_kw = {"Cancer": "all",
                       "Tissue": "all",
                       "Cell type": "all",
                       "Cluster": "all",
                       "Reliability": ["Enhanced", "Approved", "Supported"]}

    raw_data_df = pd.read_csv(data_path, sep='\t')
    data_df = raw_data_df.copy()

    # select needed rows
    for c in data_df.columns:
        if c in df_query_kw and df_query_kw[c] != "all":
            data_df = data_df.loc[data_df[c].isin(df_query_kw[c]), :]

    return {"data_df": data_df,
            "gene_names": list(set(data_df[gene_col].to_list()))}


def list_bigg_model(url="http://bigg.ucsd.edu/api/v2/models"):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data["results"])
        # Code here will only run if the request is successful
    except requests.exceptions.HTTPError as errh:
        print(errh)
    except requests.exceptions.ConnectionError as errc:
        print(errc)
    except requests.exceptions.Timeout as errt:
        print(errt)
    except requests.exceptions.RequestException as err:
        print(err)


def download_bigg_model(model_id, file_path, format="mat"):
    url = f"http://bigg.ucsd.edu/static/models/{model_id}.{format}"
