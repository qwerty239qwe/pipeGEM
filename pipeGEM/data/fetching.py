import warnings
from pathlib import Path
from typing import Dict, List, Union
import pkgutil
import hashlib
import re

import cobra.io
import requests

import numpy as np
import pandas as pd
import zeep.helpers
from zeep.exceptions import TransportError
from zeep import Client
from biodbs.HPA import HPAdb


_ORGANISM_DICT = {"human": "Homo sapiens", "mouse": "Mus musculus"}
_ORGANISM_KEGG = {"human": "hsa", "mouse": "mmu"}
_ORGANISM_BRENDA = {"human": "Homo sapiens"}


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


def _fetch_individual_kegg_gene(gene_id):
    url = f"http://rest.kegg.jp/get/{gene_id}"
    resp = requests.get(url)

    content_dic = {}
    header = None
    body = []
    text = resp.text.split("\n")
    for line in text:
        if re.match(r"\s+.*", line):
            body.append(line.lstrip())
        else:
            matched = re.match(r".*?\s+", line)
            if matched:
                if header is not None:
                    if header == "BRITE":
                        content_dic[header] = body
                    elif header == "DBLINKS":
                        for b in body:
                            k, v = b.split(": ")
                            content_dic[k] = v
                    else:
                        content_dic[header] = ";".join(body)
                header = line[matched.start(): matched.end()].strip()
                body = [line[matched.end():]]
    content_dic["KEGG ID"] = gene_id
    return content_dic


def fetch_KEGG_gene_list(organism) -> pd.DataFrame:
    if organism in _ORGANISM_KEGG:
        organism = _ORGANISM_KEGG[organism]

    kegg_data_path = pkgutil.get_data("", f"./data/kegg/{organism}.csv")

    if kegg_data_path is not None:
        return pd.read_csv(kegg_data_path)
    url = "http://rest.kegg.jp/list/{org}".format(org=organism)
    resp = requests.get(url)
    data = [line.split("\t") for line in resp.text.split("\n")]
    df = pd.DataFrame(data)
    resource_path = Path(__file__).parent.parent.parent / "data/kegg"
    try:
        df.to_csv(resource_path / f"{organism}.csv")
    except FileNotFoundError as e:
        warnings.warn(f"The kegg file couldn't be saved in the {resource_path.resolve()}")

    return df


def fetch_KEGG_gene_data(organism) -> pd.DataFrame:
    gene_list = fetch_KEGG_gene_list(organism)
    gene_data = []
    for i in range(gene_list.shape[0]):
        gene_id = gene_list.iloc[i, 0]
        ind_data = _fetch_individual_kegg_gene(gene_id)
        gene_data.append(ind_data)
    return pd.DataFrame(gene_data)


def fetch_brenda_data(account, pwd, organism, field):
    wsdl = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"
    password = hashlib.sha256(pwd.encode("utf-8")).hexdigest()
    client = Client(wsdl)

    field_methods = {"KM": ("getEcNumbersFromKmValue", "getKmValue"),
                     "MW": ("getEcNumbersFromMolecularWeight", "getMolecularWeight"),
                     "PATH": ("getEcNumbersFromPathway", "getPathway"),
                     "SEQ": ("getEcNumbersFromSequence", "getSequence"),
                     "SA": ("getEcNumbersFromSpecificActivity", "getSpecificActivity"),
                     "KCAT": ("getEcNumbersFromTurnoverNumber", "getTurnoverNumber"),}
    param_dic = {"KM": ("kmValue*", "kmValueMaximum*", "substrate*", "substrate*", "commentary*",
                         f"organism*{_ORGANISM_BRENDA[organism]}", "ligandStructureId*", "literature*"),
                 "MW": ("molecularWeight*", "molecularWeightMaximum*", "commentary*",
                         f"organism*{_ORGANISM_BRENDA[organism]}", "literature*"),
                 "PATH": ("pathway*", "link*", "source_database*", "sourceDatabase*"),
                 "SEQ": ("sequence*", "noOfAminoAcids*", "firstAccessionCode*",
                         "source*", "id*", f"organism*{_ORGANISM_BRENDA[organism]}"),
                 "SA": ("specificActivity*", "specificActivityMaximum*", "commentary*",
                        f"organism*{_ORGANISM_BRENDA[organism]}", "literature*"),
                 "KCAT": ("turnoverNumber*", "turnoverNumberMaximum*", "substrate*", "commentary*",
                          f"organism*{_ORGANISM_BRENDA[organism]}", "ligandStructureId*", "literature*"),}

    parameters = (account, password)

    field_ec_list = getattr(client.service, field_methods[field][0])(*parameters)
    results = []
    for i, ec in enumerate(field_ec_list):
        ec_parameters = (*parameters, f"ecNumber*{ec}", *param_dic[field])
        try:
            results.append(zeep.helpers.serialize_object(getattr(client.service, field_methods[field][1])(*ec_parameters)))
        except TransportError:
            warnings.warn(f"cannot get the information of {ec}")
        if i % 100 == 0:
            print(f"{i} / {len(field_ec_list)}")
    return results


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
                organism=None,
                max_n_rxns=np.inf,
                max_n_mets=np.inf,
                max_n_genes=np.inf,
                **kwargs):
    fetchers = DataBaseFetcherIniter(**kwargs)
    fetchers.register("BiGG", BiggDataBaseFetcher)
    fetchers.register("metabolic atlas", AtlasDataBaseFetcher)
    all_dfs = []
    for database in databases:
        df = fetchers.init_fetcher(database).fetch_data()
        df["database"] = database
        all_dfs.append(df)
    mg_df = pd.concat(all_dfs, axis=0)
    if organism is not None:
        organism = _format_organism_name(organism)
        mg_df = mg_df[mg_df["organism"].str.contains(organism)]
    mg_df = mg_df.query(f"gene_count <= {max_n_genes} and "
                        f"reaction_count <= {max_n_rxns} and "
                        f"metabolite_count <= {max_n_mets}")
    return mg_df


def load_model(model_id):
    model_list = list_models()
    if model_id in model_list[model_list["database"]=="BiGG"]["id"].to_list():
        return cobra.io.load_model(model_id)
    raise NotImplementedError("haven't finished")


def download_model(model_id, file_path, format="mat"):
    url = f"http://bigg.ucsd.edu/static/models/{model_id}.{format}"
    raise NotImplementedError("haven't finished")