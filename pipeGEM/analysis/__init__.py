from pipeGEM.analysis._flux import flux_analyzers, ProblemAnalyzer, modified_pfba, add_mod_pfba, add_norm_constraint
from pipeGEM.analysis._dim_reduction import prepare_PCA_dfs, prepare_embedding_dfs
from pipeGEM.analysis._stat import *
from pipeGEM.analysis.results import *
from pipeGEM.analysis._threshold import *
from pipeGEM.analysis._problem import *
from pipeGEM.analysis._gapsplit import gapsplit
from pipeGEM.analysis._reducing import *
from pipeGEM.analysis._mapping import *
from pipeGEM.analysis._ko import *
from pipeGEM.analysis.lp import *
from pipeGEM.analysis._consistency import *
from pipeGEM.analysis.scaling import *


THRESHOLD = [
    "threshold_finders",
    "LocalThresholdAnalysis",
    "PercentileThresholdAnalysis",
    "rFASTCORMICSThresholdAnalysis",
]

INTEGRATIONS = [
    "rFASTCORMICSAnalysis",
    "EFluxAnalysis",
    "MBA_Analysis",
    "CORDA_Analysis",
    "mCADRE_Analysis",
    "iMAT_Analysis",
    "FASTCOREAnalysis",
    "GIMMEAnalysis",
    "RIPTiDeSamplingAnalysis",
    "RIPTiDePruningAnalysis",
    "INIT_Analysis"
]

SCALINGS = [
    "model_scaler_collection",
    "L1NormScaler",
    "L2NormScaler",
    "deBuchetScalerP2",
    "deBuchetScalerP1",
    "ArithmeticScaler",
    "GeoMeanScaler",
    "ModelScalingResult"
]

FLUX_ANALYSIS = [
    "modified_pfba",
    "add_mod_pfba",
    "add_norm_constraint",
    "flux_analyzers",
    "FluxAnalysis",
    "FBA_Analysis",
    "FVA_Analysis",
    "FVAConsistencyAnalysis",
    "SamplingAnalysis",
]

KO_ANALYSIS = [
    "Single_KO_Analysis",
]

DIM_REDUCTIONS = [
    "PCA_Analysis",
    "EmbeddingAnalysis",
    "prepare_PCA_dfs",
    "prepare_embedding_dfs",
]

STATS = [
    "NumInequalityStoppingCriteria",
    "NormalityTestResult",
    "VarHomogeneityTestResult",
    "PairwiseTestResult"
]

CONSISTENCY = [
    "consistency_testers",
    "FastCCAnalysis",
    "FVAConsistencyAnalysis",
    "IsInSetStoppingCriteria"
]

MAPPING = [
    "RxnMapper",
]

GROUP_COMP = [
    "ComponentComparisonAnalysis",
    "ComponentNumberAnalysis",
]

__all__ = tuple(
    THRESHOLD + INTEGRATIONS + FLUX_ANALYSIS + SCALINGS + KO_ANALYSIS + DIM_REDUCTIONS + STATS +
    CONSISTENCY + MAPPING + GROUP_COMP + [
           "ProblemAnalyzer",
           "gapsplit",
           "TaskAnalysis",
           "CorrelationAnalysis",
           ]
)