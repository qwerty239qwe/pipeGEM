try:
    import torch
except ImportError:
    raise ImportError("Running DLKcat need pytorch and rdkit installed.")


try:
    import rdkit
except ImportError:
    raise ImportError("Running DLKcat need rdkit installed.")


from pipeGEM.extensions.DLKcat.model import *
from pipeGEM.extensions.DLKcat.preprocess import *
from pipeGEM.extensions.DLKcat.utils import *
from pipeGEM.extensions.DLKcat.data import *
from pipeGEM.extensions.DLKcat.main import predict_Kcat


