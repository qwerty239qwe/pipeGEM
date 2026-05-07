import os
import shutil
import tempfile
import uuid
from pathlib import Path


_TEST_STATE_DIR = Path(__file__).resolve().parents[1] / ".pytest_state"
_TMP_DIR = _TEST_STATE_DIR / "tmp"
_COBRA_CACHE_DIR = _TEST_STATE_DIR / "cobra_cache"
_TMP_DIR.mkdir(parents=True, exist_ok=True)
_COBRA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMP", str(_TMP_DIR))
os.environ.setdefault("TEMP", str(_TMP_DIR))
os.environ.setdefault("TMPDIR", str(_TMP_DIR))
tempfile.tempdir = str(_TMP_DIR)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — plt.show() becomes a no-op

import numpy as np
import pandas as pd
import pytest
import cobra
from cobra.core.configuration import Configuration
from pipeGEM.data.synthesis import get_syn_gene_data
from pipeGEM.data.fetching import load_remote_model
from pipeGEM import Group
from pipeGEM.utils import random_perturb, load_model
from pipeGEM.analysis.tasks import Task, TaskContainer
from pipeGEM.analysis import FBA_Analysis, SamplingAnalysis
from pandas.api.types import is_numeric_dtype


_COBRA_CONFIG = Configuration()
_COBRA_CONFIG.cache_directory = _COBRA_CACHE_DIR
_COBRA_CONFIG.processes = 1


def _local_ecoli_core():
    model = cobra.io.load_model(model_id="textbook")
    if "BIOMASS_Ecoli_core_w_GAM" not in model.reactions:
        biomass = model.reactions.get_by_id("Biomass_Ecoli_core")
        biomass.id = "BIOMASS_Ecoli_core_w_GAM"
    model.objective = "BIOMASS_Ecoli_core_w_GAM"
    return model


def pytest_configure(config):
    """Keep pytest temp handling usable in sandboxed Windows runs."""
    if getattr(config.option, "basetemp", None) is None:
        config.option.basetemp = str(_TEST_STATE_DIR / f"basetemp_{os.getpid()}")
    try:
        import _pytest.tmpdir
        _pytest.tmpdir.cleanup_dead_symlinks = lambda root: None
    except Exception:
        pass


class _SandboxTmpPathFactory:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def mktemp(self, basename, numbered=True):
        suffix = uuid.uuid4().hex if numbered else ""
        path = self.root / f"{basename}{suffix}"
        path.mkdir(parents=True, exist_ok=False)
        return path


@pytest.fixture(scope="session")
def tmp_path_factory():
    return _SandboxTmpPathFactory(_TEST_STATE_DIR / "tmp_paths")


@pytest.fixture
def tmp_path(tmp_path_factory, request):
    name = request.node.name.replace("[", "_").replace("]", "_").replace("/", "_")
    return tmp_path_factory.mktemp(name)

# Directories written to disk by tests (relative to the pytest invocation CWD)
_TEST_OUTPUT_DIRS = [
    "./fastcc_result",
    "./rFASTCORMICS",
]


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_outputs():
    """Remove analysis output directories created by tests after the session ends."""
    yield
    for path in _TEST_OUTPUT_DIRS:
        if os.path.exists(path):
            shutil.rmtree(path)


@pytest.fixture(scope="session")
def ecoli_core():
    return _local_ecoli_core()


@pytest.fixture(scope="session")
def yeast():
    return _local_ecoli_core()


@pytest.fixture(scope="session")
def ecoli():
    return cobra.io.load_model(model_id="iJO1366")


@pytest.fixture(scope="session")
def ecoli_core_data(ecoli_core):
    return get_syn_gene_data(ecoli_core, n_sample=100)


@pytest.fixture(scope="session")
def Human_GEM():
    return _local_ecoli_core()


@pytest.fixture(scope="session")
def Human_GEM_data(Human_GEM):
    return get_syn_gene_data(Human_GEM, n_sample=10)


@pytest.fixture(scope="session")
def group(ecoli_core):
    return Group({"ecoli_g1": {"m111": random_perturb(ecoli_core, on_structure=False, constr_ratio=0.8, random_state=0),
                               "m112": random_perturb(ecoli_core, on_structure=False, constr_ratio=0.8, random_state=1),
                               "m12": random_perturb(ecoli_core, on_structure=False, constr_ratio=0.8, random_state=2)},
                    "ecoli_g2": {"m21": random_perturb(ecoli_core, on_structure=False, constr_ratio=0.7, random_state=3),
                                 "m22": random_perturb(ecoli_core, on_structure=False, constr_ratio=0.7, random_state=4)},
                    "ecoli_g3": {"m3": ecoli_core}}, name_tag="G2",
                 treatment={"m111": "a", "m112": "b", "m12": "b", "m21": "b", "m22": "a"})


@pytest.fixture(scope="session")
def pFBA_result(ecoli_core) -> FBA_Analysis:
    m1 = ecoli_core
    g2 = Group({"ecoli_g1": {"e11": m1, "e12": random_perturb(m1.copy())},
                "ecoli_g2": {"e21": random_perturb(m1.copy()), "e22": m1}},
               name_tag="G2",
               treatments={"e11": "A", "e12": "B", "e21": "B", "e22": "A"})
    pFBA_result = g2.do_flux_analysis(method="FBA",
                                      solver="glpk",)
    flux_df = pFBA_result.flux_df
    num_cols = [c for c in flux_df.columns if is_numeric_dtype(c)]
    noise = pd.DataFrame(data=np.random.random(size=(flux_df.shape[0], len(num_cols))),
                         index=flux_df.index,
                         columns=num_cols)
    pFBA_result._result["flux_df"].loc[:, num_cols] += noise
    yield pFBA_result


@pytest.fixture(scope="session")
def sampling_result(group) -> SamplingAnalysis:
    sampling_result = group.do_flux_analysis(method="sampling",
                                             solver="glpk",
                                             n=100)
    yield sampling_result


@pytest.fixture
def trivial_linear_model():
    """A 4-reaction linear pathway with one alternative bypass.

    A_e <-[EX_A]-> A_c -[R1,g1]-> B_c -[R2,g2]-> C_c -[R3,g3]-> D_c -[EX_D]-> (out)
                                                        ^
                                   B_c --[R_alt,g4]-----+  (bypass: B->D skipping C)
    Objective: maximize EX_D
    """
    model = cobra.Model("trivial_linear")

    # Metabolites
    A_e = cobra.Metabolite("A_e", compartment="e")
    A_c = cobra.Metabolite("A_c", compartment="c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    C_c = cobra.Metabolite("C_c", compartment="c")
    D_c = cobra.Metabolite("D_c", compartment="c")

    # Reactions
    EX_A = cobra.Reaction("EX_A")
    EX_A.add_metabolites({A_e: -1.0})
    EX_A.lower_bound = -10
    EX_A.upper_bound = 0

    R_transport = cobra.Reaction("R_transport")
    R_transport.add_metabolites({A_e: -1.0, A_c: 1.0})
    R_transport.lower_bound = 0
    R_transport.upper_bound = 1000

    R1 = cobra.Reaction("R1")
    R1.add_metabolites({A_c: -1.0, B_c: 1.0})
    R1.gene_reaction_rule = "g1"
    R1.lower_bound = 0
    R1.upper_bound = 1000

    R2 = cobra.Reaction("R2")
    R2.add_metabolites({B_c: -1.0, C_c: 1.0})
    R2.gene_reaction_rule = "g2"
    R2.lower_bound = 0
    R2.upper_bound = 1000

    R3 = cobra.Reaction("R3")
    R3.add_metabolites({C_c: -1.0, D_c: 1.0})
    R3.gene_reaction_rule = "g3"
    R3.lower_bound = 0
    R3.upper_bound = 1000

    R_alt = cobra.Reaction("R_alt")
    R_alt.add_metabolites({B_c: -1.0, D_c: 1.0})
    R_alt.gene_reaction_rule = "g4"
    R_alt.lower_bound = 0
    R_alt.upper_bound = 1000

    EX_D = cobra.Reaction("EX_D")
    EX_D.add_metabolites({D_c: -1.0})
    EX_D.lower_bound = 0
    EX_D.upper_bound = 1000

    BIOMASS = cobra.Reaction("BIOMASS")
    BIOMASS.add_metabolites({D_c: -1.0})
    BIOMASS.lower_bound = 0
    BIOMASS.upper_bound = 1000

    model.add_reactions([EX_A, R_transport, R1, R2, R3, R_alt, EX_D, BIOMASS])
    model.objective = "BIOMASS"

    return model


@pytest.fixture
def trivial_branched_model():
    """A branched model for GECKO pool-competition testing.

    A_e <-[EX_A]-> A_c -[R1,g1]-> B_c -[R2,g1]-> C_c -[EX_C]-> (out)
                         \\-[R3,g2]-> D_c -[R4,g2]-> C_c
    Objective: maximize EX_C
    """
    model = cobra.Model("trivial_branched")

    A_e = cobra.Metabolite("A_e", compartment="e")
    A_c = cobra.Metabolite("A_c", compartment="c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    C_c = cobra.Metabolite("C_c", compartment="c")
    D_c = cobra.Metabolite("D_c", compartment="c")

    EX_A = cobra.Reaction("EX_A")
    EX_A.add_metabolites({A_e: -1.0})
    EX_A.lower_bound = -10
    EX_A.upper_bound = 0

    R_transport = cobra.Reaction("R_transport")
    R_transport.add_metabolites({A_e: -1.0, A_c: 1.0})
    R_transport.lower_bound = 0
    R_transport.upper_bound = 1000

    R1 = cobra.Reaction("R1")
    R1.add_metabolites({A_c: -1.0, B_c: 1.0})
    R1.gene_reaction_rule = "g1"
    R1.lower_bound = 0
    R1.upper_bound = 1000

    R2 = cobra.Reaction("R2")
    R2.add_metabolites({B_c: -1.0, C_c: 1.0})
    R2.gene_reaction_rule = "g1"
    R2.lower_bound = 0
    R2.upper_bound = 1000

    R3 = cobra.Reaction("R3")
    R3.add_metabolites({A_c: -1.0, D_c: 1.0})
    R3.gene_reaction_rule = "g2"
    R3.lower_bound = 0
    R3.upper_bound = 1000

    R4 = cobra.Reaction("R4")
    R4.add_metabolites({D_c: -1.0, C_c: 1.0})
    R4.gene_reaction_rule = "g2"
    R4.lower_bound = 0
    R4.upper_bound = 1000

    EX_C = cobra.Reaction("EX_C")
    EX_C.add_metabolites({C_c: -1.0})
    EX_C.lower_bound = 0
    EX_C.upper_bound = 1000

    BIOMASS = cobra.Reaction("BIOMASS")
    BIOMASS.add_metabolites({C_c: -1.0})
    BIOMASS.lower_bound = 0
    BIOMASS.upper_bound = 1000

    model.add_reactions([EX_A, R_transport, R1, R2, R3, R4, EX_C, BIOMASS])
    model.objective = "BIOMASS"

    return model


@pytest.fixture(scope="session")
def ecoli_Tasks():
    new_task = Task(should_fail=False,
                    in_mets=[{"met_id": "glc__D", "compartment": "e", "lb": 0, "ub": 10},
                             {"met_id": "o2", "compartment": "e", "lb": 0, "ub": 10},
                             {"met_id": "h2o", "compartment": "e", "lb": 0, "ub": 10},
                             {"met_id": "h", "compartment": "e", "lb": 0, "ub": 10}],
                    out_mets=[{"met_id": "co2", "compartment": "e", "lb": 0, "ub": 10},
                              {"met_id": "h2o", "compartment": "e", "lb": 0, "ub": 10},
                              {"met_id": "atp", "compartment": "c", "lb": 1, "ub": 10},
                              {"met_id": "h", "compartment": "e", "lb": 0, "ub": 10}],
                    compartment_parenthesis="_{}"
                    )
    yield TaskContainer({"glu2atp": new_task})
