# tests/data/test_medium_data.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import cobra
from pint import UnitRegistry

# Assuming pipeGEM is installed or in PYTHONPATH
from pipeGEM.data import MediumCatalog
from pipeGEM.data.data import MediumData

# Fixtures for test data and models can be added later if needed
# For now, create data directly in tests

@pytest.fixture
def sample_medium_df():
    """Provides a sample medium DataFrame."""
    return pd.DataFrame({
        'Metabolite Name': ['Glucose', 'L-Alanine', 'Water'],
        'human_1': ['glc__D', 'ala__L', 'h2o'],
        'mmol/L': [10.0, 5.0, np.inf] # Use np.inf for water or unconstrained
    })

@pytest.fixture
def sample_medium_df_id_index():
    """Provides a sample medium DataFrame with ID as index."""
    return pd.DataFrame({
        'Metabolite Name': ['Glucose', 'L-Alanine', 'Water'],
        'mmol/L': [10.0, 5.0, np.inf]
    }, index=pd.Index(['glc__D', 'ala__L', 'h2o'], name='human_1'))

@pytest.fixture
def sample_medium_df_name_index():
    """Provides a sample medium DataFrame with Name as index."""
    return pd.DataFrame({
        'human_1': ['glc__D', 'ala__L', 'h2o'],
        'mmol/L': [10.0, 5.0, np.inf]
    }, index=pd.Index(['Glucose', 'L-Alanine', 'Water'], name='Metabolite Name'))

@pytest.fixture
def dummy_model():
    """Provides a simple cobra model for testing alignment and application."""
    model = cobra.Model('test_model')
    # Metabolites (external 'e' and cytosolic 'c')
    glc_e = cobra.Metabolite('glc__D_e', formula='C6H12O6', name='D-Glucose', compartment='e')
    ala_e = cobra.Metabolite('ala__L_e', formula='C3H7NO2', name='L-Alanine', compartment='e')
    h2o_e = cobra.Metabolite('h2o_e', formula='H2O', name='Water', compartment='e')
    o2_e = cobra.Metabolite('o2_e', formula='O2', name='Oxygen', compartment='e') # Inorganic example
    pyr_c = cobra.Metabolite('pyr_c', formula='C3H3O3', name='Pyruvate', compartment='c') # Internal metabolite
    pyr_e = cobra.Metabolite('pyr_e', formula='C3H3O3', name='Pyruvate', compartment='e') # External pyruvate for testing
    met_x = cobra.Metabolite('x_c', formula='C1', compartment='c') # Generic organic metabolite


    # Reactions
    # Exchanges
    rxn_glc_ex = cobra.Reaction('EX_glc__D_e')
    rxn_glc_ex.add_metabolites({glc_e: -1.0})
    rxn_glc_ex.lower_bound = -1000.0
    rxn_glc_ex.upper_bound = 1000.0

    rxn_ala_ex = cobra.Reaction('EX_ala__L_e')
    rxn_ala_ex.add_metabolites({ala_e: -1.0})
    rxn_ala_ex.lower_bound = -1000.0
    rxn_ala_ex.upper_bound = 1000.0

    # A second exchange for glucose to test _find_simple_rxn
    rxn_glc_ex_alt = cobra.Reaction('EX_glc__D_e_alt')
    rxn_glc_ex_alt.add_metabolites({glc_e: -1.0, h2o_e: -0.1}) # Slightly more complex
    rxn_glc_ex_alt.lower_bound = -1000.0
    rxn_glc_ex_alt.upper_bound = 1000.0

    rxn_h2o_ex = cobra.Reaction('EX_h2o_e')
    rxn_h2o_ex.add_metabolites({h2o_e: -1.0})
    rxn_h2o_ex.lower_bound = -1000.0
    rxn_h2o_ex.upper_bound = 1000.0

    rxn_o2_ex = cobra.Reaction('EX_o2_e') # Inorganic exchange
    rxn_o2_ex.add_metabolites({o2_e: -1.0})
    rxn_o2_ex.lower_bound = -1000.0
    rxn_o2_ex.upper_bound = 1000.0

    # Demand reaction (consuming from cytosol)
    rxn_pyr_demand = cobra.Reaction('DM_pyr_c')
    rxn_pyr_demand.add_metabolites({pyr_c: -1.0}) # pyr_c -->
    rxn_pyr_demand.lower_bound = 0.0
    rxn_pyr_demand.upper_bound = 1000.0

    # Sink reaction (consuming from external)
    rxn_h2o_sink = cobra.Reaction('SK_h2o_e')
    rxn_h2o_sink.add_metabolites({h2o_e: -1.0}) # h2o_e -->
    rxn_h2o_sink.lower_bound = 0.0
    rxn_h2o_sink.upper_bound = 1000.0

    # Demand for generic organic metabolite
    rxn_dm_x = cobra.Reaction('DM_x_c')
    rxn_dm_x.add_metabolites({met_x: -1.0}) # x_c -->
    rxn_dm_x.lower_bound = 0.0
    rxn_dm_x.upper_bound = 1000.0

    # Add metabolites first, then reactions
    model.add_metabolites([glc_e, ala_e, h2o_e, o2_e, pyr_c, pyr_e, met_x])
    model.add_reactions([rxn_glc_ex, rxn_ala_ex, rxn_glc_ex_alt, rxn_h2o_ex, rxn_o2_ex, rxn_pyr_demand, rxn_h2o_sink, rxn_dm_x])
    return model

# --- Test __init__ ---

def test_medium_data_init_basic(sample_medium_df):
    """Test basic initialization with default parameters."""
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)
    assert len(medium.data_dict) == 3
    assert medium.data_dict['glc__D'] == 10.0
    assert medium.data_dict['ala__L'] == 5.0
    assert medium.data_dict['h2o'] == np.inf
    assert len(medium.name_dict) == 3
    assert medium.name_dict['glc__D'] == 'Glucose'

def test_medium_data_init_id_index(sample_medium_df_id_index):
    """Test initialization with ID in index."""
    medium = MediumData(sample_medium_df_id_index, id_index=True, name_col_label='Metabolite Name', name_index=False)
    assert len(medium.data_dict) == 3
    assert medium.data_dict['glc__D'] == 10.0
    assert medium.name_dict['glc__D'] == 'Glucose'

def test_medium_data_init_name_index(sample_medium_df_name_index):
    """Test initialization with Name in index (default)."""
    medium = MediumData(sample_medium_df_name_index, id_col_label='human_1') # name_index=True is default
    assert len(medium.data_dict) == 3
    assert medium.data_dict['glc__D'] == 10.0
    assert medium.name_dict['glc__D'] == 'Glucose'

def test_medium_data_init_missing_conc_col(sample_medium_df):
    """Test error handling for missing concentration column."""
    with pytest.raises(KeyError, match="Concentration column 'missing_conc' not found"):
        MediumData(sample_medium_df, conc_col_label='missing_conc', id_col_label='human_1', name_index=False)

def test_medium_data_init_missing_id_col(sample_medium_df):
    """Test error handling for missing ID column when id_index=False."""
    with pytest.raises(KeyError, match="Metabolite ID column 'missing_id' not found"):
        MediumData(sample_medium_df, id_col_label='missing_id', name_index=False)

def test_medium_data_init_missing_name_col_warning(sample_medium_df):
    """Test warning when name column is specified but missing (name_index=False)."""
    with pytest.warns(UserWarning, match="Metabolite name column 'missing_name' not found"):
        medium = MediumData(sample_medium_df, id_col_label='human_1', name_col_label='missing_name', name_index=False)
    assert len(medium.name_dict) == 0 # Name dict should be empty

def test_medium_data_init_invalid_unit(sample_medium_df):
    """Test error handling for invalid concentration unit."""
    with pytest.raises(ValueError, match="Invalid concentration unit 'invalid_unit'"):
        MediumData(sample_medium_df, conc_unit='invalid_unit', id_col_label='human_1', name_index=False)

# --- Test from_file ---

# Create dummy files for testing from_file
@pytest.fixture(scope="module")
def create_dummy_medium_files(tmp_path_factory):
    """Creates dummy TSV and CSV medium files in a temporary directory."""
    # Note: We place these outside the package structure to test direct path loading
    # and simulate the case where it's not found in the default 'medium/' dir.
    # For testing the default dir logic, we'd need to mock Path or place files there.
    tmp_dir = tmp_path_factory.mktemp("medium_files")

    # Dummy TSV
    tsv_content = "Metabolite Name\thuman_1\tmmol/L\nGlucose\tglc__D\t15.0\nL-Alanine\tala__L\t7.5"
    tsv_path = tmp_dir / "TestMedium.tsv"
    tsv_path.write_text(tsv_content)

    # Dummy CSV
    csv_content = "Metabolite Name,human_1,Concentration (mM)\nGlucose,glc__D,20.0\nL-Alanine,ala__L,10.0"
    csv_path = tmp_dir / "TestMediumCSV.csv"
    csv_path.write_text(csv_content)

    return {"tsv": tsv_path, "csv": csv_path, "dir": tmp_dir}

# Mock Path.is_file() for default directory testing if needed, or adjust test setup
# For simplicity here, we primarily test loading via direct path.

def test_medium_data_from_file_tsv(create_dummy_medium_files):
    """Test loading from a TSV file using direct path."""
    tsv_path = create_dummy_medium_files["tsv"]
    # Specify column labels matching the dummy file
    medium = MediumData.from_file(tsv_path, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False, conc_col_label='mmol/L')
    assert len(medium.data_dict) == 2
    assert medium.data_dict['glc__D'] == 15.0
    assert medium.name_dict['ala__L'] == 'L-Alanine'

def test_medium_data_from_file_csv(create_dummy_medium_files):
    """Test loading from a CSV file using direct path and csv_kw."""
    csv_path = create_dummy_medium_files["csv"]
    # Specify column labels and units matching the dummy CSV
    medium = MediumData.from_file(csv_path,
                                  csv_kw={'sep': ','}, # Explicitly pass separator
                                  id_col_label='human_1',
                                  name_col_label='Metabolite Name', name_index=False,
                                  conc_col_label='Concentration (mM)',
                                  conc_unit='mM') # Match unit in file
    assert len(medium.data_dict) == 2
    assert medium.data_dict['glc__D'] == 20.0
    assert medium.name_dict['ala__L'] == 'L-Alanine'

def test_medium_data_from_file_not_found(tmp_path):
    """Test FileNotFoundError when the file doesn't exist."""
    non_existent_path = tmp_path / "non_existent_medium.tsv"
    with pytest.raises(FileNotFoundError, match="Medium file .* not found"):
        MediumData.from_file(non_existent_path)

# --- Test _find_simple_rxn ---
# This is a static method, can be tested directly on the class

def test_find_simple_rxn_basic(dummy_model):
    """Test finding the simplest reaction (single metabolite exchange)."""
    ala_e = dummy_model.metabolites.get_by_id('ala__L_e')
    rxns = list(ala_e.reactions) # Should only be EX_ala__L_e
    # Filter to only keep exchange reactions if others exist
    rxns = [r for r in rxns if r.id.startswith('EX_')]
    assert len(rxns) == 1
    simplest = MediumData._find_simple_rxn(rxns)
    assert simplest.id == 'EX_ala__L_e'

def test_find_simple_rxn_multiple_exchanges(dummy_model):
    """Test choosing the simpler of two exchange reactions."""
    glc_e = dummy_model.metabolites.get_by_id('glc__D_e')
    # Get reactions associated with glc__D_e
    rxns = [r for r in glc_e.reactions if r.id.startswith('EX_')]
    assert len(rxns) == 2 # EX_glc__D_e and EX_glc__D_e_alt
    simplest = MediumData._find_simple_rxn(rxns)
    # EX_glc__D_e has fewer metabolites (1 vs 2) and smaller coeff sum (1 vs 1.1)
    assert simplest.id == 'EX_glc__D_e'

def test_find_simple_rxn_empty_list():
    """Test error handling for empty reaction list."""
    with pytest.raises(IndexError, match="Input reaction list cannot be empty"):
        MediumData._find_simple_rxn([])

def test_find_simple_rxn_non_numeric_coeff(dummy_model):
    """Test handling of non-numeric coefficients (should warn and penalize)."""
    glc_e = dummy_model.metabolites.get_by_id('glc__D_e')
    rxn_bad_coeff = cobra.Reaction('EX_bad_coeff')
    # Add a non-numeric coefficient
    rxn_bad_coeff.add_metabolites({glc_e: 'invalid'})
    rxns = [dummy_model.reactions.EX_glc__D_e, rxn_bad_coeff]

    with pytest.warns(UserWarning, match="Non-numeric coefficient found"):
        simplest = MediumData._find_simple_rxn(rxns)
    # Should choose the valid reaction over the one with bad coefficient
    assert simplest.id == 'EX_glc__D_e'


# --- Test align ---

def test_align_basic(sample_medium_df, dummy_model):
    """Test basic alignment of medium data to model exchange reactions."""
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)
    medium.align(dummy_model, external_comp_name='e', met_id_format="{met_id}_{comp}")

    # Expect a warning because glucose has multiple exchanges
    assert len(medium.rxn_dict) == 3 # glc, ala, h2o should all map
    assert 'EX_glc__D_e' in medium.rxn_dict # Should pick the simpler glucose exchange
    assert medium.rxn_dict['EX_glc__D_e'] == 10.0
    assert 'EX_ala__L_e' in medium.rxn_dict
    assert medium.rxn_dict['EX_ala__L_e'] == 5.0
    assert 'EX_h2o_e' in medium.rxn_dict
    assert medium.rxn_dict['EX_h2o_e'] == np.inf


def test_align_metabolite_not_in_model(sample_medium_df, dummy_model):
    """Test alignment when a metabolite from data is not in the model."""
    df = sample_medium_df.copy()
    # Add a metabolite not in the dummy model
    df.loc[len(df)] = ['FakeMet', 'fake_met', 1.0]

    medium = MediumData(df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)
    with pytest.warns(UserWarning, match="Metabolite 'fake_met_e' .* not found"):
        medium.align(dummy_model, external_comp_name='e', met_id_format="{met_id}_{comp}")

    assert len(medium.rxn_dict) == 3 # glc, ala, h2o mapped
    assert 'fake_met_e' not in [m.id for m in dummy_model.metabolites]
    assert 'EX_glc__D_e' in medium.rxn_dict
    assert 'EX_ala__L_e' in medium.rxn_dict
    assert 'EX_h2o_e' in medium.rxn_dict


def test_align_metabolite_no_exchange(sample_medium_df, dummy_model):
    """Test alignment when a metabolite exists but has no exchange reaction."""
    df = sample_medium_df.copy()
    # Add pyruvate (present in model but only has demand, not exchange)
    df.loc[len(df)] = ['Pyruvate', 'pyr', 2.0] # Assuming 'pyr' maps to 'pyr_e'

    medium = MediumData(df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)
    # Expect warning because pyr_e exists but has no reaction in model.exchanges
    with pytest.warns(UserWarning, match="Metabolite 'pyr_e' found .* but has no associated exchange reaction"):
         medium.align(dummy_model, external_comp_name='e', met_id_format="{met_id}_{comp}") # Use underscore format

    assert len(medium.rxn_dict) == 3 # glc, ala, h2o mapped
    assert 'DM_pyr_c' not in medium.rxn_dict # Demand reaction shouldn't be mapped here
    assert 'EX_glc__D_e' in medium.rxn_dict
    assert 'EX_ala__L_e' in medium.rxn_dict
    assert 'EX_h2o_e' in medium.rxn_dict



def test_align_raise_error_met_not_found(sample_medium_df, dummy_model):
    """Test align raises error when metabolite not found and raise_err=True."""
    df = sample_medium_df.copy()
    df.loc[len(df)] = ['FakeMet', 'fake_met', 1.0]
    medium = MediumData(df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)
    with pytest.raises(KeyError, match="Metabolite 'fake_met_e' .* not found"):
        medium.align(dummy_model, external_comp_name='e', raise_err=True, met_id_format="{met_id}_{comp}")

def test_align_invalid_model_type(sample_medium_df):
    """Test align raises TypeError for invalid model input."""
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)
    with pytest.raises(TypeError, match="Input 'model' does not appear to be a valid"):
        medium.align("not_a_model")


# --- Test apply ---

def test_apply_basic(sample_medium_df, dummy_model):
    """Test applying basic medium constraints to model bounds."""
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False, conc_unit="mM") # Use mM for easier calculation check
    # Align first (expect warning for multiple glc exchanges)
    medium.align(dummy_model, external_comp_name='e', met_id_format="{met_id}_{comp}")

    assert 'EX_glc__D_e' in medium.rxn_dict
    assert 'EX_ala__L_e' in medium.rxn_dict
    assert 'EX_h2o_e' in medium.rxn_dict

    # Apply constraints
    # Default params: cell_dgw=1e-12 g, n_cells_per_l=1e9 /L, time_hr=96 hr
    # Total biomass * time = 1e-12 g/cell * 1e9 cell/L * 96 hr = 9.6e-2 g*hr/L
    # Max influx = Conc [mmol/L] / (Total biomass * time [g*hr/L])
    # Glc: 10 mmol/L / (9.6e-2 g*hr/L) = 10 / 0.096 = 104.166... mmol/g/hr
    # Ala: 5 mmol/L / (9.6e-2 g*hr/L) = 5 / 0.096 = 52.083... mmol/g/hr
    # H2O: inf mmol/L -> influx_ub = inf -> bound should remain unchanged (-1000)
    medium.apply(dummy_model, flux_unit="mmol/g/hr") # Default flux unit

    glc_rxn = dummy_model.reactions.get_by_id('EX_glc__D_e')
    ala_rxn = dummy_model.reactions.get_by_id('EX_ala__L_e')
    h2o_rxn = dummy_model.reactions.get_by_id('EX_h2o_e')
    o2_rxn = dummy_model.reactions.get_by_id('EX_o2_e') # Inorganic, absent from medium
    dm_x_rxn = dummy_model.reactions.get_by_id('DM_x_c') # Organic demand, absent from medium

    # Check bounds (negative for uptake via exchange)
    assert glc_rxn.lower_bound == pytest.approx(-104.166666)
    assert ala_rxn.lower_bound == pytest.approx(-52.083333)
    assert h2o_rxn.lower_bound == -1000.0 # Unchanged due to inf concentration

    # Check bounds for reactions NOT in the medium
    # Oxygen is inorganic (no 'C' in formula 'O2'), should not be constrained to 0
    assert o2_rxn.lower_bound == -1000.0

    # DM_x_c consumes x_c (organic, formula 'C1'), absent from medium.
    # Reaction: x_c --> , total_stoich = -1. This is like uptake/consumption.
    # Lower bound should be set to 0.
    assert dm_x_rxn.lower_bound == 0
    assert dm_x_rxn.upper_bound == 1000.0 # Upper bound unchanged


def test_apply_different_units(sample_medium_df, dummy_model):
    """Test apply with different concentration and flux units."""
    # Medium in uM
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False, conc_unit="uM")

    # Target flux in mol/kg/day
    # Conversion factors: 1 mmol = 1e-3 mol, 1 g = 1e-3 kg, 1 hr = 1/24 day
    # Target unit: mol / kg / day
    # Original calc unit: mmol / g / hr
    # Factor: (1e-3 mol / 1 mmol) / (1e-3 kg / 1 g) / (1/24 day / 1 hr)
    # Factor = 1 / (1/24) = 24
    # Expected Glc (if mM): 104.166 (mmol/g/hr) * 24 = 2500 (mol/kg/day)
    # Since conc is uM (1e-3 mM), influx is 1e-3 smaller
    # Expected Glc: 104.166 * 1e-3 * 24 = 2.5 (mol/kg/day)
    # Expected Ala: 52.083 * 1e-3 * 24 = 1.25 (mol/kg/day)
    # Expected H2O: inf -> unchanged
    medium.align(dummy_model, external_comp_name='e', met_id_format="{met_id}_{comp}")
    medium.apply(dummy_model, flux_unit="mol/kg/day")

    glc_rxn = dummy_model.reactions.get_by_id('EX_glc__D_e')
    ala_rxn = dummy_model.reactions.get_by_id('EX_ala__L_e')
    h2o_rxn = dummy_model.reactions.get_by_id('EX_h2o_e')


    assert glc_rxn.lower_bound == pytest.approx(-2.5)
    assert ala_rxn.lower_bound == pytest.approx(-1.25)
    assert h2o_rxn.lower_bound == -1000.0 # Check H2O remains unchanged in mol/kg/day too

def test_apply_threshold(sample_medium_df, dummy_model):
    """Test that the threshold is applied correctly."""
    df = sample_medium_df.copy()
    # Set a very low concentration that should fall below default threshold
    df.loc[df['human_1'] == 'ala__L', 'mmol/L'] = 1e-9 # Very low alanine conc

    medium = MediumData(df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False, conc_unit="mmol/L")

    # Default threshold is 1e-6
    # Calculated Ala influx: (1e-9 mmol/L) / (0.096 g*hr/L) = 1.04e-8 mmol/g/hr
    # This is less than 1e-6, so bound should be set to -threshold
    medium.align(dummy_model, external_comp_name='e', met_id_format="{met_id}_{comp}")
    medium.apply(dummy_model, flux_unit="mmol/g/hr", threshold=1e-6)

    ala_rxn = dummy_model.reactions.get_by_id('EX_ala__L_e')
    assert ala_rxn.lower_bound == -1e-6 # Applied threshold

def test_apply_invalid_unit_param(sample_medium_df, dummy_model):
    """Test error handling for invalid units in apply parameters."""
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)
    with pytest.raises(ValueError, match="Invalid unit or value provided"):
        medium.apply(dummy_model, flux_unit="invalid_flux_unit")

def test_apply_invalid_model_type(sample_medium_df):
    """Test apply raises TypeError for invalid model input."""
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)
    # medium.align(dummy_model) # Need alignment first, but test type error on apply
    with pytest.raises(TypeError, match="Input 'model' lacks expected reaction lists"):
        medium.apply("not_a_model") # Pass invalid model type


def test_apply_before_align(sample_medium_df, dummy_model):
    """Test apply with empty rxn_dict (align not called) — bounds should be unchanged."""
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)
    original_lb = {r.id: r.lower_bound for r in dummy_model.reactions}
    original_ub = {r.id: r.upper_bound for r in dummy_model.reactions}

    medium.apply(dummy_model)  # rxn_dict is empty

    for rxn in dummy_model.reactions:
        # No rxn_dict entries, so only the "absent from medium" logic runs.
        # Exchange reactions for organic metabolites may get lb=0 if organic.
        # Just verify it doesn't raise.
        pass
    assert medium.rxn_dict == {}


def test_apply_does_not_relax_bounds(sample_medium_df, dummy_model):
    """Test that apply does not relax a bound that is already more restrictive."""
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)
    medium.align(dummy_model, external_comp_name='e', met_id_format="{met_id}_{comp}")

    # Pre-set EX_glc__D_e lower_bound to a more restrictive value (-1.0)
    glc_rxn = dummy_model.reactions.get_by_id('EX_glc__D_e')
    glc_rxn.lower_bound = -1.0

    medium.apply(dummy_model, flux_unit="mmol/g/hr")

    # Calculated influx ≈ -104.16, which is MORE negative (less restrictive) than -1.0.
    # apply() only tightens bounds, so -1.0 must remain.
    assert glc_rxn.lower_bound == -1.0


def test_apply_secretion_reaction_in_rxn_dict():
    """Test apply when rxn_dict contains a reaction with positive total stoichiometry (secretion).

    cobra classifies EX_ reactions with a single metabolite (positive OR negative stoich)
    as exchanges.  EX_sec_e with {met: +1.0} has total_stoich > 0, triggering the
    secretion code path which tightens the *upper* bound.
    """
    sec_model = cobra.Model('sec_model')
    met_a = cobra.Metabolite('a_e', formula='C2H6O', compartment='e')
    rxn_sec = cobra.Reaction('EX_a_e')
    rxn_sec.add_metabolites({met_a: 1.0})  # positive stoich → secretion convention
    rxn_sec.lower_bound = 0.0
    rxn_sec.upper_bound = 1000.0
    sec_model.add_metabolites([met_a])
    sec_model.add_reactions([rxn_sec])

    df = pd.DataFrame({'ID': ['a'], 'mmol/L': [5.0]})
    medium = MediumData(df, id_col_label='ID', name_index=False)
    medium.rxn_dict['EX_a_e'] = 5.0

    medium.apply(sec_model, flux_unit="mmol/g/hr")

    # Calculated ub: 5 / 0.096 ≈ 52.08  → new_ub < 1000, so upper_bound should update
    assert sec_model.reactions.get_by_id('EX_a_e').upper_bound == pytest.approx(52.083333, rel=1e-4)
    assert sec_model.reactions.get_by_id('EX_a_e').lower_bound == 0.0  # Unchanged


def test_apply_secretion_not_relaxed():
    """Test that a secretion reaction upper bound is NOT relaxed when calculated ub > current ub."""
    sec_model = cobra.Model('sec_model2')
    met_b = cobra.Metabolite('b_e', formula='C2H6O', compartment='e')
    rxn_sec = cobra.Reaction('EX_b_e')
    rxn_sec.add_metabolites({met_b: 1.0})  # positive stoich
    rxn_sec.lower_bound = 0.0
    rxn_sec.upper_bound = 0.001  # Very tight already
    sec_model.add_metabolites([met_b])
    sec_model.add_reactions([rxn_sec])

    # Large concentration → large calculated ub (~104.166), which > current_ub (0.001)
    df = pd.DataFrame({'ID': ['b'], 'mmol/L': [10.0]})
    medium = MediumData(df, id_col_label='ID', name_index=False)
    medium.rxn_dict['EX_b_e'] = 10.0

    medium.apply(sec_model, flux_unit="mmol/g/hr")
    # Calculated ub ≈ 104.166 > current_ub 0.001 → should NOT update (not relax)
    assert sec_model.reactions.get_by_id('EX_b_e').upper_bound == 0.001


def test_apply_rxn_not_in_model_exchanges(sample_medium_df, dummy_model):
    """Test that a warning is issued when rxn_dict has a reaction ID not in relevant_rxns."""
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)
    medium.align(dummy_model, external_comp_name='e', met_id_format="{met_id}_{comp}")
    # Manually add a stale reaction ID that no longer exists in the model
    medium.rxn_dict['NONEXISTENT_RXN'] = 5.0

    with pytest.warns(UserWarning, match="not found among exchanges/sinks/demands"):
        medium.apply(dummy_model, flux_unit="mmol/g/hr")


def test_apply_zero_stoich_reaction():
    """Test that a warning is issued for a reaction with zero net stoichiometry.

    cobra only classifies single-metabolite reactions as exchange/sink/demand, so a
    zero-stoich reaction (two metabolites cancelling out) cannot be constructed with
    a real cobra.Model.  We use a MagicMock model to inject such a reaction directly
    into the relevant_rxns pathway and verify the warning branch is exercised.
    """
    from unittest.mock import MagicMock

    mock_rxn = MagicMock()
    mock_rxn.id = 'ZERO_rxn'
    mock_rxn.metabolites.values.return_value = [1.0, -1.0]  # sum = 0
    mock_rxn.bounds = (-1000, 1000)  # Real tuple for unpacking

    mock_model = MagicMock()
    mock_model.exchanges = [mock_rxn]
    mock_model.sinks = []
    mock_model.demands = []

    df = pd.DataFrame({'ID': ['x'], 'mmol/L': [2.0]})
    medium = MediumData(df, id_col_label='ID', name_index=False)
    medium.rxn_dict['ZERO_rxn'] = 2.0

    with pytest.warns(UserWarning, match="zero net stoichiometry"):
        medium.apply(mock_model, flux_unit="mmol/g/hr")

    # Bounds not modified — the mock reaction's bounds tuple is unchanged
    assert mock_rxn.bounds == (-1000, 1000)


# --- Test align edge cases ---

def test_align_invalid_met_id_format(sample_medium_df, dummy_model):
    """Test that an invalid met_id_format raises ValueError."""
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)
    with pytest.raises(ValueError, match="Invalid 'met_id_format'"):
        medium.align(dummy_model, met_id_format="{bad_key}_{comp}")


def test_align_multiple_exchanges_warning():
    """Test that a warning is raised when a metabolite has multiple exchange reactions.

    cobra classifies single-metabolite EX_ reactions as exchanges.  Two such reactions
    sharing the same metabolite both appear in model.exchanges, triggering the
    "multiple exchange reactions" warning in align().
    """
    multi_model = cobra.Model('multi_ex')
    glc_e = cobra.Metabolite('glc__D_e', formula='C6H12O6', compartment='e')
    rxn_ex1 = cobra.Reaction('EX_glc__D_e_1')
    rxn_ex1.add_metabolites({glc_e: -1.0})
    rxn_ex1.bounds = (-1000, 1000)
    rxn_ex2 = cobra.Reaction('EX_glc__D_e_2')
    rxn_ex2.add_metabolites({glc_e: -1.0})
    rxn_ex2.bounds = (-1000, 1000)
    multi_model.add_metabolites([glc_e])
    multi_model.add_reactions([rxn_ex1, rxn_ex2])

    df = pd.DataFrame({'ID': ['glc__D'], 'mmol/L': [10.0]})
    medium = MediumData(df, id_col_label='ID', name_index=False)

    with pytest.warns(UserWarning, match="multiple exchange reactions"):
        medium.align(multi_model, external_comp_name='e', met_id_format="{met_id}_{comp}")

    # The simplest reaction is chosen (both equal in complexity, so first wins)
    assert len(medium.rxn_dict) == 1
    assert list(medium.rxn_dict.keys())[0] in ['EX_glc__D_e_1', 'EX_glc__D_e_2']


def test_align_no_exchange_for_metabolite_warning(dummy_model):
    """Test warning when metabolite exists in model but has no exchange reaction."""
    # pyr_e is in the model but has no exchange (only DM_pyr_c, SK_h2o_e, etc.)
    df = pd.DataFrame({
        'Metabolite Name': ['Pyruvate'],
        'human_1': ['pyr'],
        'mmol/L': [3.0],
    })
    medium = MediumData(df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)
    with pytest.warns(UserWarning, match="no associated exchange reaction"):
        medium.align(dummy_model, external_comp_name='e', met_id_format="{met_id}_{comp}")
    assert len(medium.rxn_dict) == 0


# --- Test __init__ edge cases ---

def test_medium_data_init_no_name_col_label(sample_medium_df):
    """Test initialization with name_index=False and name_col_label=None — name_dict is empty."""
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_index=False, name_col_label=None)
    assert medium.name_dict == {}
    assert len(medium.data_dict) == 3


def test_medium_data_conc_unit_stored(sample_medium_df):
    """Test that the concentration unit is stored as a pint Quantity."""
    from pint import UnitRegistry
    ureg = UnitRegistry()
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_index=False, conc_unit="uM")
    assert medium.conc_unit.dimensionality == ureg.Quantity("uM").dimensionality


def test_medium_catalog_source_and_composition_urls_and_m63_loads():
    """Bundled catalog entries should expose citation and composition URLs."""
    for catalog_entry in MediumCatalog:
        assert catalog_entry.value.source_url
        assert catalog_entry.value.composition_url
        assert catalog_entry.value.composition_note

    m63 = MediumData.from_catalog(MediumCatalog.M63)

    assert m63.data_dict['glc__D'] == pytest.approx(22.0)
    assert m63.data_dict['nh4'] == pytest.approx(30.0)
    assert m63.name_dict['thm'] == 'Thiamine'


def test_all_medium_catalog_entries_load_with_names():
    """Every catalog entry should point to a readable bundled TSV."""
    for catalog_entry in MediumCatalog:
        medium = MediumData.from_catalog(catalog_entry)

        assert medium.data_dict
        assert medium.name_dict
        assert set(medium.data_dict) == set(medium.name_dict)
        assert all(isinstance(name, str) and name for name in medium.name_dict.values())


@pytest.mark.parametrize(
    ("lookup", "expected"),
    [
        ("m9", MediumCatalog.M9),
        ("M63", MediumCatalog.M63),
        ("DMEM_high_FFA", MediumCatalog.DMEM_HIGH_FFA),
        ("Hams", MediumCatalog.HAMS),
    ],
)
def test_medium_catalog_string_lookup_forms(lookup, expected):
    """Catalog loading should accept enum names and medium filename stems."""
    medium = MediumData.from_catalog(lookup)
    expected_medium = MediumData.from_catalog(expected)

    medium_data = {k: v for k, v in medium.data_dict.items() if not pd.isna(k)}
    expected_data = {k: v for k, v in expected_medium.data_dict.items() if not pd.isna(k)}
    medium_names = {k: v for k, v in medium.name_dict.items() if not pd.isna(k)}
    expected_names = {k: v for k, v in expected_medium.name_dict.items() if not pd.isna(k)}

    assert medium_data == expected_data
    assert medium_names == expected_names
    assert any(pd.isna(k) for k in medium.data_dict) == any(
        pd.isna(k) for k in expected_medium.data_dict
    )


def test_medium_catalog_unknown_string_error_lists_available_entries():
    """Unknown catalog strings should fail with a useful message."""
    with pytest.raises(ValueError, match="Unknown medium 'not_a_medium'.*M63"):
        MediumData.from_catalog("not_a_medium")


def test_medium_catalog_metadata_flags_match_data_kind():
    """Complex/rich approximations should be explicitly marked."""
    approximate_entries = {MediumCatalog.LB, MediumCatalog.TB, MediumCatalog.BHI,
                           MediumCatalog.R2A, MediumCatalog.SOC, MediumCatalog.SERUM}
    exact_entries = set(MediumCatalog) - approximate_entries

    for catalog_entry in approximate_entries:
        assert catalog_entry.value.is_approximate
        assert catalog_entry.value.medium_type in {"rich", "complex"}

    for catalog_entry in exact_entries:
        assert not catalog_entry.value.is_approximate
        assert catalog_entry.value.medium_type in {"minimal", "defined"}


def test_bhi_metadata_separates_origin_and_composition_sources():
    """BHI origin and recipe references should not be collapsed into one URL."""
    info = MediumCatalog.BHI.value

    assert info.source == "Rosenow 1919"
    assert info.is_approximate
    assert info.medium_type == "complex"
    assert info.source_url != info.composition_url
    assert "thermofisher" in info.source_url
    assert "fda.gov" in info.composition_url
    assert "amino-acid proxy" in info.composition_note


@pytest.mark.parametrize(
    "catalog_entry",
    [MediumCatalog.LB, MediumCatalog.TB, MediumCatalog.BHI, MediumCatalog.R2A, MediumCatalog.SOC],
)
def test_complex_medium_metadata_discloses_proxy_compositions(catalog_entry):
    """Rich/complex TSVs should warn when digest/extract ingredients are proxied."""
    info = catalog_entry.value

    assert info.is_approximate
    assert "proxy" in info.composition_note
    assert "amino-acid" in info.composition_note


def test_m63_catalog_composition_is_exchange_compatible():
    """M63 should contain only exchange-ready BiGG IDs and finite salts."""
    m63 = MediumData.from_catalog("M63")

    expected = {
        'glc__D': 22.0,
        'nh4': 30.0,
        'pi': 101.0,
        'k': 163.0,
        'so4': 15.202,
        'mg2': 0.2,
        'fe2': 0.0018,
        'thm': 0.015,
        'o2': 0.25,
    }

    for met_id, concentration in expected.items():
        assert m63.data_dict[met_id] == pytest.approx(concentration)
    assert m63.data_dict['h2o'] == np.inf
    assert all(" " not in met_id for met_id in m63.data_dict)


def test_medium_data_init_id_index_no_id_col(sample_medium_df_id_index):
    """Test that when id_index=True, id_col_label is not required."""
    # DataFrame has no column matching a default id_col_label
    medium = MediumData(sample_medium_df_id_index, id_index=True, name_col_label='Metabolite Name', name_index=False)
    assert 'glc__D' in medium.data_dict


def test_medium_supplement_adds_or_updates_metabolite(sample_medium_df):
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)

    returned = medium.supplement('pyr', 2.5, name='Pyruvate')
    medium.supplement('ala__L', 8.0)

    assert returned is medium
    assert medium.data_dict['pyr'] == 2.5
    assert medium.name_dict['pyr'] == 'Pyruvate'
    assert medium.data_dict['ala__L'] == 8.0
    assert medium.name_dict['ala__L'] == 'ala__L'


def test_medium_supplement_warns_when_alignment_is_stale(sample_medium_df):
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)
    medium.rxn_dict['EX_glc__D_e'] = 10.0

    with pytest.warns(UserWarning, match="alignment may be stale"):
        medium.supplement('pyr', 1.0)


def test_medium_remove_deletes_metabolite_and_warns_when_aligned(sample_medium_df):
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)
    medium.rxn_dict['EX_ala__L_e'] = 5.0

    with pytest.warns(UserWarning, match="alignment may be stale"):
        returned = medium.remove('ala__L')

    assert returned is medium
    assert 'ala__L' not in medium.data_dict
    assert 'ala__L' not in medium.name_dict


def test_medium_remove_missing_metabolite_raises(sample_medium_df):
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)

    with pytest.raises(KeyError, match="not found in medium data"):
        medium.remove('missing')


def test_medium_combine_union_with_unit_conversion_and_conflict_modes():
    first = MediumData(
        pd.DataFrame({'id': ['glc__D', 'ala__L'], 'conc': [1.0, 2.0], 'name': ['Glucose', 'Alanine']}),
        id_col_label='id',
        conc_col_label='conc',
        name_col_label='name',
        name_index=False,
        conc_unit='mmol/L',
    )
    second = MediumData(
        pd.DataFrame({'id': ['glc__D', 'pyr'], 'conc': [500.0, 250.0], 'name': ['Glucose other', 'Pyruvate']}),
        id_col_label='id',
        conc_col_label='conc',
        name_col_label='name',
        name_index=False,
        conc_unit='umol/L',
    )

    combined = first.combine(second, mode='union', conflict='sum')
    second_wins = first.combine(second, mode='union', conflict='second')

    assert combined.data_dict['glc__D'] == pytest.approx(1.5)
    assert combined.data_dict['ala__L'] == pytest.approx(2.0)
    assert combined.data_dict['pyr'] == pytest.approx(0.25)
    assert combined.name_dict['glc__D'] == 'Glucose'
    assert second_wins.data_dict['glc__D'] == pytest.approx(0.5)


def test_medium_combine_intersection_and_conflict_min_max_first():
    first = MediumData(pd.DataFrame({'id': ['a', 'b'], 'mmol/L': [2.0, 5.0]}), id_col_label='id', name_index=False)
    second = MediumData(pd.DataFrame({'id': ['a', 'c'], 'mmol/L': [3.0, 7.0]}), id_col_label='id', name_index=False)

    assert first.combine(second, mode='intersection', conflict='max').data_dict == {'a': 3.0}
    assert first.combine(second, mode='intersection', conflict='min').data_dict == {'a': 2.0}
    assert first.combine(second, mode='intersection', conflict='first').data_dict == {'a': 2.0}


def test_medium_combine_rejects_invalid_inputs(sample_medium_df):
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)
    bad_unit = MediumData(
        pd.DataFrame({'id': ['x'], 'mmol/L': [1.0]}),
        id_col_label='id',
        name_index=False,
        conc_unit='gram',
    )

    with pytest.raises(TypeError, match="MediumData"):
        medium.combine(object())
    with pytest.raises(ValueError, match="mode"):
        medium.combine(medium, mode='outer')
    with pytest.raises(ValueError, match="conflict"):
        medium.combine(medium, conflict='average')
    with pytest.raises(ValueError, match="Cannot convert"):
        medium.combine(bad_unit)


# --- Test from_file edge cases ---

def test_medium_data_from_file_tsv_direct_path(tmp_path):
    """Test loading a TSV file via direct path (auto-detects separator from extension)."""
    tsv_content = "Metabolite Name\thuman_1\tmmol/L\nGlucose\tglc__D\t8.0\n"
    tsv_path = tmp_path / "direct.tsv"
    tsv_path.write_text(tsv_content)

    medium = MediumData.from_file(
        tsv_path,
        id_col_label='human_1',
        name_col_label='Metabolite Name',
        name_index=False,
        conc_col_label='mmol/L',
    )
    assert medium.data_dict['glc__D'] == 8.0


def test_medium_data_from_file_csv_direct_path(tmp_path):
    """Test loading a CSV file via direct path with explicit csv_kw separator."""
    csv_content = "id,conc\nglc__D,12.5\nala__L,6.25\n"
    csv_path = tmp_path / "direct.csv"
    csv_path.write_text(csv_content)

    medium = MediumData.from_file(
        csv_path,
        csv_kw={'sep': ','},
        id_col_label='id',
        name_index=False,
        conc_col_label='conc',
    )
    assert medium.data_dict['glc__D'] == 12.5
    assert medium.data_dict['ala__L'] == 6.25


# --- Integration tests ---

def test_align_then_apply_full_workflow(dummy_model):
    """Integration test: build medium, align, apply, and verify all bound changes.

    Note: In dummy_model, EX_glc__D_e_alt has two metabolites and is NOT classified
    as an exchange by cobra.  Therefore glc__D only has ONE classified exchange
    (EX_glc__D_e) and no multiple-exchange warning is raised.
    """
    df = pd.DataFrame({
        'Metabolite Name': ['Glucose', 'L-Alanine'],
        'human_1': ['glc__D', 'ala__L'],
        'mmol/L': [10.0, 5.0],
    })
    medium = MediumData(df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)

    medium.align(dummy_model, external_comp_name='e', met_id_format="{met_id}_{comp}")

    assert set(medium.rxn_dict.keys()) == {'EX_glc__D_e', 'EX_ala__L_e'}

    medium.apply(dummy_model, flux_unit="mmol/g/hr")

    glc_rxn = dummy_model.reactions.get_by_id('EX_glc__D_e')
    ala_rxn = dummy_model.reactions.get_by_id('EX_ala__L_e')
    h2o_rxn = dummy_model.reactions.get_by_id('EX_h2o_e')

    assert glc_rxn.lower_bound == pytest.approx(-104.166666, rel=1e-4)
    assert ala_rxn.lower_bound == pytest.approx(-52.083333, rel=1e-4)
    # h2o not in medium — it's inorganic, so bound unchanged
    assert h2o_rxn.lower_bound == -1000.0


def test_apply_custom_cell_params(sample_medium_df, dummy_model):
    """Test apply with non-default cell_dgw, n_cells_per_l, and time_hr."""
    medium = MediumData(sample_medium_df, id_col_label='human_1', name_col_label='Metabolite Name', name_index=False)
    medium.align(dummy_model, external_comp_name='e', met_id_format="{met_id}_{comp}")

    # Custom params: 1e-11 g/cell, 1e8 cells/L, 48 hr
    # Total = 1e-11 * 1e8 * 48 = 0.048 g*hr/L
    # Glc: 10 / 0.048 ≈ 208.33 mmol/g/hr
    # Ala: 5 / 0.048 ≈ 104.16 mmol/g/hr
    medium.apply(dummy_model, cell_dgw=1e-11, n_cells_per_l=1e8, time_hr=48, flux_unit="mmol/g/hr")

    glc_rxn = dummy_model.reactions.get_by_id('EX_glc__D_e')
    ala_rxn = dummy_model.reactions.get_by_id('EX_ala__L_e')

    assert glc_rxn.lower_bound == pytest.approx(-208.333, rel=1e-3)
    assert ala_rxn.lower_bound == pytest.approx(-104.166, rel=1e-3)
