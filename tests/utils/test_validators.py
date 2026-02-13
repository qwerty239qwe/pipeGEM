"""Tests for pipeGEM/utils/_validators.py."""
import cobra
import pandas as pd
import pytest
from unittest.mock import MagicMock

from pipeGEM.utils._validators import (
    validate_cobra_model,
    validate_dataframe,
    validate_gene_ids,
    validate_reaction_ids,
)
from pipeGEM.exceptions import (
    ModelValidationError,
    DataValidationError,
    DataAlignmentError,
)


# =====================================================================
# validate_cobra_model
# =====================================================================

class TestValidateCobraModel:
    def test_valid_model_passes(self):
        model = cobra.Model("test")
        result = validate_cobra_model(model)
        assert result is model

    def test_pipegem_model_extracts_inner(self):
        """pipeGEM.Model with .cobra_model attr extracts inner model."""
        inner = cobra.Model("inner")
        wrapper = MagicMock()
        wrapper.cobra_model = inner
        result = validate_cobra_model(wrapper)
        assert result is inner

    def test_string_raises(self):
        with pytest.raises(ModelValidationError, match="cobra.Model"):
            validate_cobra_model("not a model")

    def test_none_raises(self):
        with pytest.raises(ModelValidationError):
            validate_cobra_model(None)


# =====================================================================
# validate_dataframe
# =====================================================================

class TestValidateDataframe:
    def test_valid_df_returns(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = validate_dataframe(df)
        assert result is df

    def test_dict_raises(self):
        with pytest.raises(DataValidationError, match="DataFrame"):
            validate_dataframe({"a": 1})

    def test_missing_required_columns_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(DataValidationError, match="missing required column"):
            validate_dataframe(df, required_columns=["a", "b"])

    def test_all_required_columns_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = validate_dataframe(df, required_columns=["a", "b"])
        assert result is df


# =====================================================================
# validate_gene_ids
# =====================================================================

class TestValidateGeneIds:
    @pytest.fixture
    def model(self):
        m = cobra.Model("test")
        r = cobra.Reaction("R1")
        r.gene_reaction_rule = "g1 and g2 and g3"
        m.add_reactions([r])
        return m

    def test_all_overlap_returns_all(self, model):
        result = validate_gene_ids(["g1", "g2", "g3"], model)
        assert set(result) == {"g1", "g2", "g3"}

    def test_partial_returns_subset(self, model):
        result = validate_gene_ids(["g1", "g2", "gX"], model)
        assert set(result) == {"g1", "g2"}

    def test_zero_overlap_with_min_overlap_raises(self, model):
        with pytest.raises(DataAlignmentError, match="0/2"):
            validate_gene_ids(["gX", "gY"], model, min_overlap=0.5)

    def test_empty_list_returns_empty(self, model):
        result = validate_gene_ids([], model)
        assert result == []


# =====================================================================
# validate_reaction_ids
# =====================================================================

class TestValidateReactionIds:
    @pytest.fixture
    def model(self):
        m = cobra.Model("test")
        for rid in ["R1", "R2", "R3"]:
            r = cobra.Reaction(rid)
            r.add_metabolites({cobra.Metabolite(f"m_{rid}", compartment="c"): -1})
            m.add_reactions([r])
        return m

    def test_all_overlap(self, model):
        result = validate_reaction_ids(["R1", "R2", "R3"], model)
        assert set(result) == {"R1", "R2", "R3"}

    def test_partial_overlap(self, model):
        result = validate_reaction_ids(["R1", "RX"], model)
        assert result == ["R1"]

    def test_low_overlap_raises(self, model):
        with pytest.raises(DataAlignmentError):
            validate_reaction_ids(["RX", "RY"], model, min_overlap=0.5)

    def test_empty_list(self, model):
        result = validate_reaction_ids([], model)
        assert result == []
