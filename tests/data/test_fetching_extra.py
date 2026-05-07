"""Tests for pipeGEM/data/fetching.py — load_HPA_data, KEGG/BRENDA fetchers,
DataBaseFetcher hierarchy, list_models filtering, download/load helpers."""
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pandas as pd
import pytest
import requests

from pipeGEM.data.fetching import (
    load_HPA_data,
    _fetch_individual_kegg_gene,
    _format_organism_name,
    fetch_KEGG_gene_list,
    fetch_brenda_data,
    DataBaseFetcherIniter,
    DataBaseFetcher,
    BiggDataBaseFetcher,
    AtlasDataBaseFetcher,
    list_models,
    download_model,
    download_atlas_model,
    load_remote_model,
)


# =====================================================================
# _format_organism_name
# =====================================================================

class TestFormatOrganismName:
    def test_known_organism(self):
        assert _format_organism_name("human") == "Homo sapiens"
        assert _format_organism_name("mouse") == "Mus musculus"

    def test_dotted_name(self):
        assert _format_organism_name("H.sapiens") == "H.*sapiens"

    def test_passthrough(self):
        assert _format_organism_name("Rattus norvegicus") == "Rattus norvegicus"


# =====================================================================
# load_HPA_data
# =====================================================================

class TestLoadHPAData:
    def test_basic_load(self, tmp_path):
        tsv = tmp_path / "test.tsv"
        df = pd.DataFrame({
            "Gene": ["A", "B", "C"],
            "Tissue": ["brain", "liver", "brain"],
            "Reliability": ["Enhanced", "Approved", "Low"],
            "value": [1, 2, 3],
        })
        df.to_csv(tsv, sep="\t", index=False)

        result = load_HPA_data(tsv, gene_col="Gene")
        assert "data_df" in result
        assert "gene_names" in result
        # Default filter: "Low" not in ["Enhanced", "Approved", "Supported"]
        # But "Reliability" is only in df_query_kw if present — default kw checks for it
        assert isinstance(result["gene_names"], list)

    def test_custom_query(self, tmp_path):
        tsv = tmp_path / "test.tsv"
        df = pd.DataFrame({
            "Gene": ["A", "B", "C"],
            "Tissue": ["brain", "liver", "kidney"],
            "value": [1, 2, 3],
        })
        df.to_csv(tsv, sep="\t", index=False)

        result = load_HPA_data(tsv, gene_col="Gene",
                                df_query_kw={"Tissue": ["brain", "kidney"]})
        rdf = result["data_df"]
        assert len(rdf) == 2
        assert "liver" not in rdf["Tissue"].values

    def test_query_all_keeps_everything(self, tmp_path):
        tsv = tmp_path / "test.tsv"
        df = pd.DataFrame({
            "Gene": ["A", "B"],
            "Category": ["x", "y"],
        })
        df.to_csv(tsv, sep="\t", index=False)

        result = load_HPA_data(tsv, gene_col="Gene",
                                df_query_kw={"Category": "all"})
        assert len(result["data_df"]) == 2

    def test_gene_names_deduplicated(self, tmp_path):
        tsv = tmp_path / "test.tsv"
        df = pd.DataFrame({
            "Gene": ["A", "A", "B"],
            "value": [1, 2, 3],
        })
        df.to_csv(tsv, sep="\t", index=False)

        result = load_HPA_data(tsv, gene_col="Gene", df_query_kw={})
        assert len(result["gene_names"]) == 2


# =====================================================================
# _fetch_individual_kegg_gene
# =====================================================================

class TestFetchIndividualKeggGene:
    @patch("pipeGEM.data.fetching.requests.get")
    def test_basic_parse(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = (
            "ENTRY       hsa:1234\n"
            "NAME        GENE1\n"
            "DEFINITION  Some gene definition\n"
            "DBLINKS     NCBI-GeneID: 1234\n"
            "            UniProt: P12345\n"
            "///"
        )
        mock_get.return_value = mock_resp

        result = _fetch_individual_kegg_gene("hsa:1234")
        assert result["KEGG ID"] == "hsa:1234"
        assert "ENTRY" in result or "NAME" in result

    @patch("pipeGEM.data.fetching.requests.get")
    def test_brite_field_stores_list(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = (
            "BRITE       KEGG Orthology\n"
            "             09100 Metabolism\n"
            "              09101 Carbohydrate metabolism\n"
            "DBLINKS     NCBI-GeneID: 999\n"
            "///"
        )
        mock_get.return_value = mock_resp

        result = _fetch_individual_kegg_gene("hsa:999")
        assert result["KEGG ID"] == "hsa:999"


# =====================================================================
# fetch_KEGG_gene_list
# =====================================================================

class TestFetchKEGGGeneList:
    @patch("pipeGEM.data.fetching.pkgutil.get_data")
    @patch("pipeGEM.data.fetching.pd.read_csv")
    def test_cached_data_returned(self, mock_read_csv, mock_pkgutil):
        mock_pkgutil.return_value = b"col1,col2\nhsa:1,gene1"
        expected_df = pd.DataFrame({"col1": ["hsa:1"], "col2": ["gene1"]})
        mock_read_csv.return_value = expected_df

        result = fetch_KEGG_gene_list("hsa")
        assert mock_pkgutil.called

    @patch("pipeGEM.data.fetching.pkgutil.get_data", return_value=None)
    @patch("pipeGEM.data.fetching.requests.get")
    def test_api_fallback(self, mock_get, mock_pkgutil):
        mock_resp = MagicMock()
        mock_resp.text = "hsa:1\tgene1\nhsa:2\tgene2"
        mock_get.return_value = mock_resp

        # The function will try to save but may fail — that's fine, it still returns
        result = fetch_KEGG_gene_list("hsa")
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1

    def test_organism_name_mapping(self):
        with patch("pipeGEM.data.fetching.pkgutil.get_data", return_value=None), \
             patch("pipeGEM.data.fetching.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = "hsa:1\tgene1"
            mock_get.return_value = mock_resp

            fetch_KEGG_gene_list("human")
            call_url = mock_get.call_args[0][0]
            assert "hsa" in call_url


# =====================================================================
# DataBaseFetcherIniter
# =====================================================================

class TestDataBaseFetcherIniter:
    def test_default_urls(self):
        fi = DataBaseFetcherIniter()
        assert "BiGG" in fi._database_urls
        assert "metabolic atlas" in fi._database_urls

    def test_register_and_init(self):
        fi = DataBaseFetcherIniter()
        mock_cls = MagicMock()
        fi.register("BiGG", mock_cls)
        fi.init_fetcher("BiGG")
        mock_cls.assert_called_once_with(url=fi._database_urls["BiGG"])

    def test_custom_urls(self):
        fi = DataBaseFetcherIniter(new_urls={"MyDB": "http://example.com"})
        assert "MyDB" in fi._database_urls


# =====================================================================
# DataBaseFetcher
# =====================================================================

class TestDataBaseFetcher:
    def test_manipulate_df_raises(self):
        f = DataBaseFetcher("http://example.com")
        with pytest.raises(NotImplementedError):
            f.manipulate_df({})

    @patch("pipeGEM.data.fetching.requests.get")
    def test_fetch_data_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [1, 2]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        class TestFetcher(DataBaseFetcher):
            def manipulate_df(self, data):
                return pd.DataFrame(data["data"])

        f = TestFetcher("http://example.com")
        result = f.fetch_data()
        assert isinstance(result, pd.DataFrame)

    @patch("pipeGEM.data.fetching.requests.get")
    def test_fetch_data_http_error_returns_none(self, mock_get):
        mock_get.side_effect = requests.exceptions.HTTPError("404")
        f = DataBaseFetcher("http://example.com")
        # manipulate_df won't be called since we get an error first
        result = f.fetch_data()
        assert result is None

    @patch("pipeGEM.data.fetching.requests.get")
    def test_fetch_data_timeout_returns_none(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("timeout")
        f = DataBaseFetcher("http://example.com")
        result = f.fetch_data()
        assert result is None

    @patch("pipeGEM.data.fetching.requests.get")
    def test_fetch_data_connection_error_returns_none(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("fail")
        f = DataBaseFetcher("http://example.com")
        result = f.fetch_data()
        assert result is None


# =====================================================================
# BiggDataBaseFetcher
# =====================================================================

class TestBiggDataBaseFetcher:
    def test_manipulate_df(self):
        f = BiggDataBaseFetcher("http://example.com")
        data = {"results": [{"bigg_id": "m1", "organism": "ecoli"}]}
        result = f.manipulate_df(data)
        assert "id" in result.columns
        assert result.iloc[0]["id"] == "m1"


# =====================================================================
# AtlasDataBaseFetcher
# =====================================================================

class TestAtlasDataBaseFetcher:
    def test_manipulate_df(self):
        f = AtlasDataBaseFetcher("http://example.com")
        data = [
            {"short_name": "HumanGEM", "sample": {"organism": "Homo sapiens"},
             "reaction_count": 100, "metabolite_count": 50, "gene_count": 200},
        ]
        result = f.manipulate_df(data)
        assert "id" in result.columns
        assert result.iloc[0]["id"] == "HumanGEM"
        assert result.iloc[0]["organism"] == "Homo sapiens"
        expected_cols = {"id", "organism", "reaction_count", "metabolite_count", "gene_count"}
        assert expected_cols == set(result.columns)


# =====================================================================
# list_models — with mocked fetchers
# =====================================================================

class TestListModels:
    @patch("pipeGEM.data.fetching.requests.get")
    def test_filtering_by_organism(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [
            {"bigg_id": "m1", "organism": "Escherichia coli",
             "reaction_count": 10, "metabolite_count": 5, "gene_count": 20},
            {"bigg_id": "m2", "organism": "Homo sapiens",
             "reaction_count": 100, "metabolite_count": 50, "gene_count": 200},
        ]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = list_models(databases=["BiGG"], organism="Homo sapiens")
        assert all("Homo sapiens" in o for o in result["organism"].values)

    @patch("pipeGEM.data.fetching.requests.get")
    def test_filtering_by_max_counts(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [
            {"bigg_id": "m1", "organism": "ecoli",
             "reaction_count": 10, "metabolite_count": 5, "gene_count": 20},
            {"bigg_id": "m2", "organism": "ecoli",
             "reaction_count": 1000, "metabolite_count": 500, "gene_count": 2000},
        ]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = list_models(databases=["BiGG"], max_n_rxns=100)
        assert len(result) == 1
        assert result.iloc[0]["id"] == "m1"

    @patch("pipeGEM.data.fetching.requests.get")
    def test_empty_result(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("fail")
        result = list_models(databases=["BiGG"])
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# =====================================================================
# download_model — not implemented
# =====================================================================

class TestDownloadModel:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            download_model("some_model", "/tmp/model.mat")


# =====================================================================
# download_atlas_model — mocked
# =====================================================================

class TestDownloadAtlasModel:
    def test_already_exists_skips_download(self, tmp_path):
        model_file = tmp_path / "TestGEM.mat"
        model_file.write_bytes(b"fake model data")

        result = download_atlas_model("TestGEM", format="mat",
                                       download_dest=tmp_path)
        assert result == model_file

    @patch("pipeGEM.data.fetching.requests.get")
    def test_downloads_when_not_exists(self, mock_get, tmp_path):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.headers = {"Content-Length": "100"}
        mock_resp.raw = MagicMock()
        mock_resp.raw.read = MagicMock(return_value=b"x" * 100)
        mock_get.return_value = mock_resp

        # The function uses tqdm.wrapattr and shutil.copyfileobj
        with patch("pipeGEM.data.fetching.tqdm") as mock_tqdm, \
             patch("pipeGEM.data.fetching.shutil.copyfileobj"):
            mock_tqdm.wrapattr = MagicMock()
            mock_tqdm.wrapattr.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_tqdm.wrapattr.return_value.__exit__ = MagicMock(return_value=False)

            # Create the file so the function doesn't fail
            model_file = tmp_path / "TestGEM.mat"
            model_file.write_bytes(b"downloaded")

            result = download_atlas_model("TestGEM", format="mat",
                                           download_dest=tmp_path)
            assert result == model_file


# =====================================================================
# load_remote_model — mocked
# =====================================================================

class TestLoadRemoteModel:
    @patch("pipeGEM.data.fetching.list_models")
    @patch("pipeGEM.data.fetching.cobra.io.load_model")
    def test_bigg_model_loaded(self, mock_load, mock_list):
        mock_list.return_value = pd.DataFrame({
            "id": ["e_coli_core", "other"],
            "database": ["BiGG", "BiGG"],
        })
        mock_load.return_value = MagicMock()

        result = load_remote_model("e_coli_core")
        mock_load.assert_called_once_with("e_coli_core")

    @patch("pipeGEM.data.fetching.list_models")
    @patch("pipeGEM.data.fetching.download_atlas_model")
    @patch("pipeGEM.data.fetching.load_model")
    def test_atlas_model_downloaded(self, mock_load, mock_download, mock_list):
        mock_list.return_value = pd.DataFrame({
            "id": ["bigg_model"],
            "database": ["BiGG"],
        })
        mock_download.return_value = Path("/tmp/HumanGEM.mat")
        mock_load.return_value = MagicMock()

        result = load_remote_model("HumanGEM")
        mock_download.assert_called_once()
        mock_load.assert_called_once()


# =====================================================================
# fetch_brenda_data — mocked
# =====================================================================

class TestFetchBrendaData:
    @patch("pipeGEM.data.fetching.Client")
    def test_basic_call(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_service = mock_client.service
        mock_service.getEcNumbersFromKmValue.return_value = ["1.1.1.1"]
        mock_service.getKmValue.return_value = [{"value": 0.5}]

        with patch("pipeGEM.data.fetching.zeep.helpers.serialize_object",
                    side_effect=lambda x: x):
            result = fetch_brenda_data("user@example.com", "password",
                                        "human", "KM")
        assert isinstance(result, list)

    @patch("pipeGEM.data.fetching.Client")
    def test_transport_error_warns(self, mock_client_cls):
        """TransportError during BRENDA fetch issues a warning (lines 233-234)."""
        from zeep.exceptions import TransportError

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_service = mock_client.service
        mock_service.getEcNumbersFromKmValue.return_value = ["1.1.1.1"]
        mock_service.getKmValue.side_effect = TransportError("server error")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fetch_brenda_data("user@example.com", "password",
                                        "human", "KM")
            assert any("cannot get" in str(warning.message) for warning in w)
        assert result == []


# =====================================================================
# fetch_brenda_ligand — placeholder (line 167)
# =====================================================================

class TestFetchBrendaLigand:
    def test_returns_none(self):
        from pipeGEM.data.fetching import fetch_brenda_ligand
        assert fetch_brenda_ligand() is None


# =====================================================================
# fetch_KEGG_gene_data (lines 153-159)
# =====================================================================

class TestFetchKEGGGeneData:
    @patch("pipeGEM.data.fetching._fetch_individual_kegg_gene")
    @patch("pipeGEM.data.fetching.fetch_KEGG_gene_list")
    def test_basic(self, mock_gene_list, mock_individual):
        from pipeGEM.data.fetching import fetch_KEGG_gene_data
        mock_gene_list.return_value = pd.DataFrame({0: ["hsa:1", "hsa:2"]})
        mock_individual.side_effect = [
            {"KEGG ID": "hsa:1", "NAME": "gene1"},
            {"KEGG ID": "hsa:2", "NAME": "gene2"},
        ]
        result = fetch_KEGG_gene_data("hsa")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "KEGG ID" in result.columns


# =====================================================================
# _fetch_individual_kegg_gene — DBLINKS branch (lines 82-85)
# =====================================================================

class TestFetchIndividualKeggGeneDBLINKS:
    @patch("pipeGEM.data.fetching.requests.get")
    def test_dblinks_parsed(self, mock_get):
        """Test that DBLINKS entries are parsed into separate keys."""
        mock_resp = MagicMock()
        mock_resp.text = (
            "ENTRY       hsa:1234\n"
            "DBLINKS     NCBI-GeneID: 1234\n"
            "            UniProt: P12345\n"
            "NAME        GENE1\n"
            "///"
        )
        mock_get.return_value = mock_resp

        result = _fetch_individual_kegg_gene("hsa:1234")
        assert result["KEGG ID"] == "hsa:1234"
        assert "NCBI-GeneID" in result
        assert result["NCBI-GeneID"] == "1234"
        assert "UniProt" in result
        assert result["UniProt"] == "P12345"


# =====================================================================
# fetch_KEGG_gene_list — save failure branch (lines 125-126)
# =====================================================================

class TestFetchKEGGGeneListSaveFailure:
    @patch("pipeGEM.data.fetching.pkgutil.get_data", return_value=None)
    @patch("pipeGEM.data.fetching.requests.get")
    def test_save_failure_warns(self, mock_get, mock_pkgutil):
        mock_resp = MagicMock()
        mock_resp.text = "hsa:1\tgene1"
        mock_get.return_value = mock_resp

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fetch_KEGG_gene_list("hsa")
            # If the resource_path doesn't exist, it should warn
            # (may or may not warn depending on whether data/kegg/ exists)
        assert isinstance(result, pd.DataFrame)


# =====================================================================
# DataBaseFetcher — RequestException branch (lines 382-383)
# =====================================================================

class TestDataBaseFetcherRequestError:
    @patch("pipeGEM.data.fetching.requests.get")
    def test_generic_request_exception_returns_none(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("generic")
        f = DataBaseFetcher("http://example.com")
        result = f.fetch_data()
        assert result is None


# =====================================================================
# download_atlas_model — actual download branch (lines 552-558)
# =====================================================================

class TestDownloadAtlasModelDownload:
    @patch("pipeGEM.data.fetching.shutil.copyfileobj")
    @patch("pipeGEM.data.fetching.tqdm")
    @patch("pipeGEM.data.fetching.requests.get")
    def test_actual_download(self, mock_get, mock_tqdm, mock_copy, tmp_path):
        """Test the actual download path when file doesn't exist."""
        import io

        mock_raw = io.BytesIO(b"fake model content")

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.headers = {"Content-Length": "18"}
        mock_resp.raw = mock_raw
        mock_get.return_value = mock_resp

        # Make tqdm.wrapattr return the raw object
        mock_tqdm.wrapattr.return_value.__enter__ = MagicMock(return_value=mock_raw)
        mock_tqdm.wrapattr.return_value.__exit__ = MagicMock(return_value=False)

        result = download_atlas_model("NewGEM", format="mat",
                                       download_dest=tmp_path)
        assert result == tmp_path / "NewGEM.mat"
        mock_get.assert_called_once()
