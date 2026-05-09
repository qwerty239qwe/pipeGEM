import warnings
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Union
import pkgutil
import hashlib
import re

import cobra.io
import requests
import shutil
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import zeep.helpers
from zeep.exceptions import TransportError
from zeep import Client
from biodbs.fetch import hpa_search

from pipeGEM.utils import load_model
from pipeGEM._logging import get_logger

logger = get_logger(__name__)


_ORGANISM_DICT = {"human": "Homo sapiens", "mouse": "Mus musculus"}
_ORGANISM_KEGG = {"human": "hsa", "mouse": "mmu"}
_ORGANISM_BRENDA = {"human": "Homo sapiens"}


def fetch_HPA_data(data_name: str,
                   data_path: Union[str, Path] = Path(__file__).parent.parent.parent / Path("external_data/HPA")) -> dict:
    """
    Fetch Human Protein Atlas (HPA) data.

    Downloads the specified HPA dataset if it doesn't exist locally.

    Parameters
    ----------
    data_name : str
        The name of the HPA dataset to fetch (e.g., 'rna_tissue_consensus').
    data_path : Union[str, Path], optional
        The directory path to save or load the data from.
        Defaults to 'external_data/HPA' relative to the project root.

    Returns
    -------
    dict
        A dictionary containing the path to the downloaded TSV file under the key "data_path".
    """
    if isinstance(data_path, str):
        data_path = Path(data_path)

    data_path.mkdir(parents=True, exist_ok=True)
    tsv_path = (data_path / Path(data_name)).with_suffix(".tsv")
    if not tsv_path.exists():
        logger.info("Fetching data...")
        result = hpa_search(data_name)
        result.to_csv(str(tsv_path))
    else:
        logger.info("The dataframe already exists.")
    return {"data_path": tsv_path}


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
    """Fetch the list of genes for an organism from KEGG.

    Returns a cached local copy if available; otherwise queries the
    KEGG REST API and caches the result as a CSV file.

    Parameters
    ----------
    organism : str
        Organism name (e.g. ``"human"``, ``"mouse"``) or KEGG organism
        code (e.g. ``"hsa"``, ``"mmu"``).

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame with KEGG gene IDs and gene descriptions.
    """
    if organism in _ORGANISM_KEGG:
        organism = _ORGANISM_KEGG[organism]

    kegg_data_path = pkgutil.get_data("", f"./data/kegg/{organism}.csv")

    if kegg_data_path is not None:
        return pd.read_csv(BytesIO(kegg_data_path))
    url = "http://rest.kegg.jp/list/{org}".format(org=organism)
    resp = requests.get(url)
    data = [line.split("\t") for line in resp.text.split("\n")]
    df = pd.DataFrame(data)
    resource_path = Path(__file__).parent.parent.parent / "data/kegg"
    try:
        resource_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(resource_path / f"{organism}.csv")
    except OSError:
        warnings.warn(f"The kegg file couldn't be saved in the {resource_path.resolve()}")

    return df


def fetch_KEGG_gene_data(organism) -> pd.DataFrame:
    """Fetch detailed gene data for every gene in a KEGG organism.

    Iterates over the gene list returned by :func:`fetch_KEGG_gene_list`
    and retrieves per-gene annotations (DBLINKS, BRITE hierarchy, etc.)
    from the KEGG REST API.

    .. warning::
       This function makes one HTTP request per gene and can be very slow
       for organisms with many genes.

    Parameters
    ----------
    organism : str
        Organism name or KEGG code (see :func:`fetch_KEGG_gene_list`).

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per gene and columns for each parsed KEGG
        field (e.g. ``"NAME"``, ``"DEFINITION"``, ``"KEGG ID"``).
    """
    gene_list = fetch_KEGG_gene_list(organism)
    gene_data = []
    for i in range(gene_list.shape[0]):
        gene_id = gene_list.iloc[i, 0]
        ind_data = _fetch_individual_kegg_gene(gene_id)
        gene_data.append(ind_data)
    return pd.DataFrame(gene_data)


def fetch_brenda_ligand():
    """Fetch ligand data from the BRENDA database.

    .. note:: Not yet implemented.
    """
    pass


def fetch_brenda_data(account, pwd, organism, field):
    """Fetch enzyme kinetic data from the BRENDA SOAP API.

    Queries BRENDA for a specific data field across all EC numbers for
    the given organism.  Requires a registered BRENDA account.

    Parameters
    ----------
    account : str
        BRENDA account e-mail address.
    pwd : str
        BRENDA account password (will be SHA-256 hashed before sending).
    organism : str
        Organism name key (e.g. ``"human"``).  Must exist in the
        internal ``_ORGANISM_BRENDA`` mapping.
    field : str
        Data field to retrieve.  Supported values:

        * ``"KM"`` — Michaelis constant
        * ``"MW"`` — molecular weight
        * ``"PATH"`` — pathway information
        * ``"SEQ"`` — protein sequence
        * ``"SA"`` — specific activity
        * ``"KCAT"`` — turnover number
        * ``"LIGAND"`` — ligand information

    Returns
    -------
    list[dict]
        A list of serialised SOAP response objects, one per EC number.
    """
    wsdl = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"
    password = hashlib.sha256(pwd.encode("utf-8")).hexdigest()
    client = Client(wsdl)

    field_methods = {"KM": ("getEcNumbersFromKmValue", "getKmValue"),
                     "MW": ("getEcNumbersFromMolecularWeight", "getMolecularWeight"),
                     "PATH": ("getEcNumbersFromPathway", "getPathway"),
                     "SEQ": ("getEcNumbersFromSequence", "getSequence"),
                     "SA": ("getEcNumbersFromSpecificActivity", "getSpecificActivity"),
                     "KCAT": ("getEcNumbersFromTurnoverNumber", "getTurnoverNumber"),
                     "LIGAND": ("getEcNumbersFromLigands", "getLigands",)}
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
                          f"organism*{_ORGANISM_BRENDA[organism]}", "ligandStructureId*", "literature*"),
                 "LIGAND": ()}

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
            logger.info("%d / %d", i, len(field_ec_list))
    return results


def load_HPA_data(data_path: Union[str, Path],
                  gene_col: str,
                  df_query_kw: Dict[str, Union[str, List[str]]] = None,
                  ) -> dict:
    """Load and filter a Human Protein Atlas TSV file.

    Reads a tab-separated HPA data file and filters rows according to
    *df_query_kw*.  Columns whose names appear as keys in *df_query_kw*
    are kept only where their values match the specified whitelist (or
    ``"all"`` to skip filtering for that column).

    Parameters
    ----------
    data_path : str or pathlib.Path
        Path to the HPA TSV file (e.g. downloaded by
        :func:`fetch_HPA_data`).
    gene_col : str
        Column containing gene identifiers in the data file.
    df_query_kw : dict, optional
        Filtering criteria.  Keys are column names; values are either
        ``"all"`` (keep everything) or a list of allowed values.
        Defaults to a standard filter retaining ``"Enhanced"``,
        ``"Approved"``, and ``"Supported"`` reliability entries.

    Returns
    -------
    dict
        ``{"data_df": pd.DataFrame, "gene_names": list[str]}`` —
        the filtered DataFrame and a deduplicated list of gene names.
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
    """Registry that maps database names to fetcher classes and their API URLs.

    Parameters
    ----------
    new_urls : dict, optional
        Additional ``{database_name: url}`` entries to register alongside
        the built-in BiGG and Metabolic Atlas endpoints.
    """

    _database_urls = {"BiGG": "http://bigg.ucsd.edu/api/v2/models",
                      "metabolic atlas": "https://metabolicatlas.org/api/v2/repository/integrated_models"}

    def __init__(self, new_urls=None):
        self.fetchers = {}
        if new_urls is not None:
            self._database_urls.update(new_urls)

    def register(self, name, fetcher):
        """Register a :class:`DataBaseFetcher` subclass under *name*."""
        self.fetchers[name] = fetcher

    def init_fetcher(self, name):
        """Instantiate and return the fetcher registered under *name*."""
        return self.fetchers[name](url=self._database_urls[name])


class DataBaseFetcher:
    """Base class for fetching model metadata from a remote database.

    Subclasses must implement :meth:`manipulate_df` to convert the raw
    JSON response into a standardised :class:`~pandas.DataFrame`.

    Parameters
    ----------
    url : str
        API endpoint URL.
    """

    def __init__(self, url):
        self.url = url

    def manipulate_df(self, data) -> pd.DataFrame:
        """Convert a raw JSON response into a DataFrame.

        Must be overridden by subclasses.

        Parameters
        ----------
        data : dict or list
            Parsed JSON from the API response.

        Returns
        -------
        pd.DataFrame
        """
        raise NotImplementedError()

    def fetch_data(self) -> pd.DataFrame:
        """Fetch model metadata from the remote API.

        Sends a GET request to :attr:`url`, parses the JSON response,
        and delegates to :meth:`manipulate_df` for schema normalisation.

        Returns
        -------
        pd.DataFrame or None
            Normalised DataFrame of model metadata, or ``None`` if the
            request fails (errors are logged, not raised).
        """
        try:
            headers = {
                'User-Agent': 'FETCHER'
            }
            response = requests.get(self.url, timeout=30, headers=headers)
            response.raise_for_status()
            data = response.json()
            return self.manipulate_df(data)
            # Code here will only run if the request is successful
        except requests.exceptions.HTTPError as errh:
            logger.error("HTTP error: %s", errh)
        except requests.exceptions.ConnectionError as errc:
            logger.error("Connection error: %s", errc)
        except requests.exceptions.Timeout as errt:
            logger.error("Timeout error: %s", errt)
        except requests.exceptions.RequestException as err:
            logger.error("Request error: %s", err)


class BiggDataBaseFetcher(DataBaseFetcher):
    """Fetcher for the BiGG Models database API."""

    def __init__(self, url):
        super().__init__(url=url)

    def manipulate_df(self, data):
        return pd.DataFrame(data["results"]).rename(columns={"bigg_id": "id"})


class AtlasDataBaseFetcher(DataBaseFetcher):
    """Fetcher for the Metabolic Atlas repository API."""

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
                **kwargs) -> pd.DataFrame:
    """
    List available metabolic models from specified databases with optional filtering.

    Parameters
    ----------
    databases : List[str], optional
        A list of database names to fetch models from (e.g., ["metabolic atlas", "BiGG"]).
        Defaults to ["metabolic atlas", "BiGG"].
    organism : str, optional
        Filter models by organism name (e.g., "human", "mouse"). Case-insensitive.
    max_n_rxns : float, optional
        Maximum number of reactions allowed in the models. Defaults to infinity.
    max_n_mets : float, optional
        Maximum number of metabolites allowed in the models. Defaults to infinity.
    max_n_genes : float, optional
        Maximum number of genes allowed in the models. Defaults to infinity.
    **kwargs
        Additional keyword arguments for DataBaseFetcherIniter.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing information about the available models, including
        'id', 'organism', 'reaction_count', 'metabolite_count', 'gene_count',
        and 'database'. Returns an empty DataFrame if no data is fetched.
    """
    fetchers = DataBaseFetcherIniter(**kwargs)
    fetchers.register("BiGG", BiggDataBaseFetcher)
    fetchers.register("metabolic atlas", AtlasDataBaseFetcher)
    all_dfs = []
    for database in databases:
        df = fetchers.init_fetcher(database).fetch_data()
        if df is None:
            logger.warning("Cannot fetch %s", database)
            continue

        df["database"] = database
        all_dfs.append(df)
    if len(all_dfs) == 0:
        logger.warning("No data fetched, returning an empty dataframe.")
        return pd.DataFrame()

    mg_df = pd.concat(all_dfs, axis=0)
    if organism is not None:
        organism = _format_organism_name(organism)
        mg_df = mg_df[mg_df["organism"].str.contains(organism)]
    mg_df = mg_df.query(f"gene_count <= {max_n_genes} and "
                        f"reaction_count <= {max_n_rxns} and "
                        f"metabolite_count <= {max_n_mets}")
    return mg_df


def load_remote_model(model_id,
                      format="mat",
                      branch="main",
                      download_dest="default"):
    """
    Load a metabolic model from a remote database (BiGG or Metabolic Atlas).

    If the model_id is found in the BiGG database, it is loaded directly using
    cobrapy. Otherwise, it attempts to download the model from the Metabolic
    Atlas GitHub repository.

    Parameters
    ----------
    model_id : str
        The ID of the model to load.
    format : str, optional
        The format of the model file to download (e.g., "mat", "xml", "yml").
        Defaults to "mat".
    branch : str, optional
        The GitHub branch to download the model from for Metabolic Atlas models.
        Defaults to "main".
    download_dest : str, optional
        The destination directory to download the model to. Defaults to "default",
        which saves to a 'models' directory relative to the project root.

    Returns
    -------
    cobra.Model
        The loaded metabolic model.
    """
    model_list = list_models()
    if model_id in model_list[model_list["database"]=="BiGG"]["id"].to_list():
        return cobra.io.load_model(model_id)
    model_path = download_atlas_model(model_id, format=format, branch=branch, download_dest=download_dest)
    return load_model(model_path)


def download_model(model_id, file_path, format="mat"):
    """Download a model from BiGG.

    .. note:: Not yet implemented.

    Parameters
    ----------
    model_id : str
        BiGG model identifier.
    file_path : str or Path
        Destination file path.
    format : str, optional
        File format (default ``"mat"``).
    """
    url = f"http://bigg.ucsd.edu/static/models/{model_id}.{format}"
    raise NotImplementedError("haven't finished")


def download_atlas_model(model_id="Human-GEM", format="mat", branch="main", download_dest="default") -> str:
    """Download a GEM from the Metabolic Atlas GitHub repository.

    If the model file already exists locally, the download is skipped.

    Parameters
    ----------
    model_id : str, optional
        Repository / model name (default ``"Human-GEM"``).
    format : str, optional
        File extension (default ``"mat"``).  Must match the filename in
        the repository's ``model/`` directory.
    branch : str, optional
        Git branch to download from (default ``"main"``).
    download_dest : str or Path, optional
        Destination directory.  ``"default"`` saves to
        ``<project_root>/models/<branch>/``.

    Returns
    -------
    pathlib.Path
        Path to the downloaded (or already-existing) model file.
    """
    download_dest = Path(__file__).resolve().parent.parent.parent / "models" / branch \
        if download_dest == "default" else download_dest
    download_dest.mkdir(parents=True, exist_ok=True)
    if (download_dest / f"{model_id}.{format}").is_file():
        logger.info("Model %s is already downloaded.", model_id)
        return download_dest / f"{model_id}.{format}"

    url = f"https://github.com/SysBioChalmers/{model_id}/raw/{branch}/model/{model_id}.{format}"
    with requests.get(url, stream=True) as r:
        total_length = int(r.headers.get("Content-Length"))
        with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
            with open(download_dest / f"{model_id}.{format}", 'wb') as output:
                shutil.copyfileobj(raw, output)
    return download_dest / f"{model_id}.{format}"
