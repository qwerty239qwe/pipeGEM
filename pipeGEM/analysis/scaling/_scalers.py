import numpy as np
import numpy.ma as ma

from ._base import ModelScaler, ModelScalerCollection


class ArithmeticScaler(ModelScaler):
    def __init__(self):
        super().__init__()

    def get_R(self,
              A: np.ndarray):
        R = (A != 0).astype(int).sum(axis=1) / abs(A).sum(axis=1)
        return np.diag(R)

    def get_S(self,
              A: np.ndarray):
        S = (A != 0).astype(int).sum(axis=0) / abs(A).sum(axis=0)
        return np.diag(S)


class deBuchetScalerP1(ModelScaler):
    def __init__(self):
        super().__init__()

    def get_R(self,
              A: np.ndarray):
        masked = ma.array(abs(A), mask=(A == 0))
        R = np.power((1 / masked).sum(axis=1).data / abs(A).sum(axis=1), 1/2)
        return np.diag(R)

    def get_S(self,
              A: np.ndarray):
        masked = ma.array(abs(A), mask=(A == 0))
        S = np.power((1 / masked).sum(axis=0).data / abs(A).sum(axis=0), 1/2)
        return np.diag(S)


class deBuchetScalerP2(ModelScaler):
    def __init__(self):
        super().__init__()

    def get_R(self,
              A: np.ndarray):
        masked = ma.array(abs(A), mask=(A == 0))
        R = np.power(np.power((1 / masked).data, 2).sum(axis=1) / abs(A).sum(axis=1), 1/4)
        return np.diag(R)

    def get_S(self,
              A: np.ndarray):
        masked = ma.array(abs(A), mask=(A == 0))
        S = np.power(np.power((1 / masked).data, 2).sum(axis=0) / abs(A).sum(axis=0), 1/4)
        return np.diag(S)


class GeoMeanScaler(ModelScaler):
    def __init__(self):
        super().__init__()

    def get_R(self,
              A: np.ndarray):
        masked = ma.array(abs(A), mask=(A == 0))
        R = np.power(masked.max(axis=1).data * masked.min(axis=1).data, -1/2)
        return np.diag(R)

    def get_S(self,
              A: np.ndarray):
        masked = ma.array(abs(A), mask=(A == 0))
        S = np.power(masked.max(axis=0).data * masked.min(axis=0).data, -1/2)
        return np.diag(S)


class L1NormScaler(ModelScaler):
    def __init__(self):
        super().__init__()

    def get_R(self,
              A: np.ndarray):
        masked = ma.array(abs(A), mask=(A == 0))
        R = 1 / ma.median(masked, axis=1).data
        return np.diag(R)

    def get_S(self,
              A: np.ndarray):
        masked = ma.array(abs(A), mask=(A == 0))
        S = 1 / ma.median(masked, axis=0).data
        return np.diag(S)


class L2NormScaler(ModelScaler):
    def __init__(self):
        super().__init__()

    def get_R(self,
              A: np.ndarray):
        masked = ma.array(abs(A), mask=(A == 0))
        counts = (masked != 0).astype(int).sum(axis=1)
        R = 1 / ma.power(ma.prod(masked, axis=1), 1/counts).data
        return np.diag(R)

    def get_S(self,
              A: np.ndarray):
        masked = ma.array(abs(A), mask=(A == 0))
        counts = (masked != 0).astype(int).sum(axis=0)
        S = 1 / ma.power(ma.prod(masked, axis=0), 1 / counts).data
        return np.diag(S)


model_scaler_collection = ModelScalerCollection()
model_scaler_collection.register("arithmetic", ArithmeticScaler)
model_scaler_collection.register("de_Buchet_p1", deBuchetScalerP1)
model_scaler_collection.register("de_Buchet_p2", deBuchetScalerP2)
model_scaler_collection.register("geometric_mean", GeoMeanScaler)
model_scaler_collection.register("l1_norm", L1NormScaler)
model_scaler_collection.register("l2_norm", L2NormScaler)