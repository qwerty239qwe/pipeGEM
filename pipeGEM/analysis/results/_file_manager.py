import numpy as np

from pipeGEM.utils import ObjectFactory, save_model, load_model, load_pg_model
import pandas as pd
import cobra
from pathlib import Path


class BaseFileManager:
    def __init__(self, type, default_suffix):
        self._type = type
        self._default_suffix = default_suffix

    @property
    def suffix(self):
        return self._default_suffix

    def is_valid_type(self, obj):
        return isinstance(obj, self._type)

    def write(self, obj, file_name, **kwargs):
        raise NotImplementedError()

    def read(self, file_name, **kwargs):
        raise NotImplementedError()


class FrameFileManager(BaseFileManager):
    def __init__(self):
        super(FrameFileManager, self).__init__(pd.DataFrame, default_suffix=".csv")

    def write(self, obj, file_name, **kwargs):
        if "suffix" in kwargs:
            suffix = kwargs.pop("suffix")
        else:
            suffix = self.suffix

        obj.to_csv(Path(file_name).with_suffix(suffix), **kwargs)

    def read(self, file_name, **kwargs):
        return pd.read_csv(file_name, **kwargs)


class SeriesFileManager(BaseFileManager):
    def __init__(self):
        super(SeriesFileManager, self).__init__(pd.Series, default_suffix=".csv")

    def write(self, obj: pd.Series, file_name, **kwargs):
        if "suffix" in kwargs:
            suffix = kwargs.pop("suffix")
        else:
            suffix = self.suffix

        obj.to_frame().to_csv(Path(file_name).with_suffix(suffix), **kwargs)

    def read(self, file_name, **kwargs):
        return pd.read_csv(file_name, index_col=0, **kwargs).iloc[:, 0]


class CobraModelFileManager(BaseFileManager):
    def __init__(self):
        super(CobraModelFileManager, self).__init__(cobra.Model,
                                                    default_suffix=".json")

    def write(self, obj, file_name, **kwargs):
        if "suffix" in kwargs:
            suffix = kwargs.pop("suffix")
        else:
            suffix = self.suffix
        save_model(model=obj, output_file_name=Path(file_name).with_suffix(suffix))

    def read(self, file_name, **kwargs):
        return load_model(model_file_path=file_name)


class PGModelFileManager(BaseFileManager):
    def __init__(self):
        super(PGModelFileManager, self).__init__("pipeGEM.Model",
                                                 default_suffix=".json")

    def write(self, obj, file_name, **kwargs):
        if "suffix" in kwargs:
            suffix = kwargs.pop("suffix")
        else:
            suffix = self.suffix
        obj.save_model(file_name=Path(file_name).with_suffix(suffix))

    def read(self, file_name, **kwargs):
        return load_pg_model(file_name=file_name)


class NDArrayStrFileManager(BaseFileManager):
    def __init__(self):
        super(NDArrayStrFileManager, self).__init__(np.ndarray,
                                                    default_suffix=".txt")

    def write(self, obj, file_name, **kwargs):
        if "suffix" in kwargs:
            suffix = kwargs.pop("suffix")
        else:
            suffix = self.suffix
        np.savetxt(Path(file_name).with_suffix(suffix),
                   obj,
                   fmt='%s',
                   **kwargs)

    def read(self, file_name, **kwargs):
        return np.loadtxt(file_name, dtype="str", **kwargs)


class NDArrayFloatFileManager(BaseFileManager):
    def __init__(self):
        super(NDArrayFloatFileManager, self).__init__(np.ndarray,
                                                      default_suffix=".npz")

    def write(self, obj, file_name, **kwargs):
        if "suffix" in kwargs:
            suffix = kwargs.pop("suffix")
        else:
            suffix = self.suffix
        np.savez(str(Path(file_name).with_suffix(suffix)),
                 obj,
                 **kwargs)

    def read(self, file_name, **kwargs):
        with np.load(file_name, **kwargs) as data:
            arr = data["arr_0"]
        return arr


class FileManagers(ObjectFactory):
    def __init__(self):
        super().__init__()


fmanagers = FileManagers()
fmanagers.register("pandas.DataFrame", FrameFileManager)
fmanagers.register("pandas.Series", SeriesFileManager)
fmanagers.register("cobra.Model", CobraModelFileManager)
fmanagers.register("pipeGEM.Model", PGModelFileManager)
fmanagers.register("numpy.NDArrayStr", NDArrayStrFileManager)
fmanagers.register("numpy.NDArrayFloat", NDArrayFloatFileManager)