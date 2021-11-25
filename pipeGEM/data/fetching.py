from pathlib import Path
from typing import Dict, List, Union
import requests

import pandas as pd
from biodbs.HPA import HPAdb


_ORGANISM_DICT = {"human": "Homo sapiens", "mouse": "Mus musculus"}


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


def _format_organism_name(raw: str):
    if raw in _ORGANISM_DICT:
        return _ORGANISM_DICT[raw]
    if "." in raw:
        return raw.replace(".", ".*")
    return raw


class DataBaseFetcherIniter:
    _database_urls = {"BiGG": "http://bigg.ucsd.edu/api/v2/models",
                      "metabolic atlas": "https://metabolicatlas.org/api/v2/repository/integrated_models"}

    def __init__(self, new_urls=None):
        self.fetchers = {}
        if new_urls is not None:
            self._database_urls.update(new_urls)

    def register(self, name, fetcher):
        self.fetchers[name] = fetcher

    def init_fetcher(self, name):
        return self.fetchers[name](url=self._database_urls[name])


class DataBaseFetcher:
    def __init__(self, url):
        self.url = url

    def manipulate_df(self, data) -> pd.DataFrame:
        raise NotImplementedError()

    def fetch_data(self) -> pd.DataFrame:
        try:
            response = requests.get(self.url, timeout=30)
            response.raise_for_status()
            data = response.json()
            return self.manipulate_df(data)
            # Code here will only run if the request is successful
        except requests.exceptions.HTTPError as errh:
            print(errh)
        except requests.exceptions.ConnectionError as errc:
            print(errc)
        except requests.exceptions.Timeout as errt:
            print(errt)
        except requests.exceptions.RequestException as err:
            print(err)


class BiggDataBaseFetcher(DataBaseFetcher):
    def __init__(self, url):
        super().__init__(url=url)

    def manipulate_df(self, data):
        return pd.DataFrame(data["results"]).rename(columns={"bigg_id": "id"})


class AtlasDataBaseFetcher(DataBaseFetcher):
    def __init__(self, url):
        super().__init__(url=url)

    def manipulate_df(self, data) -> pd.DataFrame:
        df = pd.DataFrame(data)
        df["organism"] = df["sample"].apply(lambda x: x["organism"])
        df = df.rename(columns={"short_name": "id"})
        return df.loc[:, ["id", "organism", "reaction_count", "metabolite_count", "gene_count"]]


def list_models(databases=["metabolic atlas", "BiGG"],
                **kwargs):
    fetchers = DataBaseFetcherIniter()
    fetchers.register("BiGG", BiggDataBaseFetcher)
    fetchers.register("metabolic atlas", AtlasDataBaseFetcher)
    all_dfs = []
    for database in databases:
        df = fetchers.init_fetcher(database).fetch_data()
        df["database"] = database
        all_dfs.append(df)

    return pd.concat(all_dfs, axis=0)


def load_model(model_id):
    model_list = list_models()


def download_model(model_id, file_path, format="mat"):
    url = f"http://bigg.ucsd.edu/static/models/{model_id}.{format}"
