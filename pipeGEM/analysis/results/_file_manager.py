import numpy as np

from pipeGEM.utils import ObjectFactory, save_model, load_model
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


class CSVFileManager(BaseFileManager):
    def __init__(self):
        super(CSVFileManager, self).__init__(pd.DataFrame, default_suffix=".csv")

    def write(self, obj, file_name, **kwargs):
        if "suffix" in kwargs:
            suffix = kwargs.pop("suffix")
        else:
            suffix = self.suffix

        obj.to_csv(Path(file_name).with_suffix(suffix), **kwargs)

    def read(self, file_name, **kwargs):
        return pd.read_csv(file_name, **kwargs)


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


class NDArrayStrFileManager(BaseFileManager):
    def __init__(self):
        super(NDArrayStrFileManager, self).__init__(np.ndarray,
                                                    default_suffix=".txt")

    def write(self, obj, file_name, **kwargs):
        if "suffix" in kwargs:
            suffix = kwargs.pop("suffix")
        else:
            suffix = self.suffix
        np.save_txt(Path(file_name).with_suffix(suffix),
                    obj,
                    **kwargs)

    def read(self, file_name, **kwargs):
        return np.loadtxt(file_name, **kwargs)


class FileManagers(ObjectFactory):
    def __init__(self):
        super().__init__()


fmanagers = FileManagers()
fmanagers.register(type(pd.DataFrame), CSVFileManager)
fmanagers.register(type(cobra.Model), CobraModelFileManager)
fmanagers.register("NDArrayStr", NDArrayStrFileManager)