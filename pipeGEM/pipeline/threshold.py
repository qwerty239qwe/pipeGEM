from typing import Optional

import numpy as np
import pandas as pd

from ._base import Pipeline
from pipeGEM.integration.utils import get_rfastcormics_thresholds


class BimodalThreshold(Pipeline):
    def __init__(self,
                 cut_off: float = -np.inf,
                 naming_format: Optional[str] = "./thresholds/{sample_name}.png",
                 plot_dist: bool = True,
                 use_first_guess: bool = False,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.cut_off = cut_off
        self.plot_dist = plot_dist
        self.naming_format = naming_format
        self.use_first_guess = use_first_guess

    def run(self,
            data: pd.Series,
            sample_name: str) -> (float, float):
        self.output = {}
        self.output[sample_name] = get_rfastcormics_thresholds(data.values,
                                                               cut_off=self.cut_off,
                                                               file_name=self.naming_format.format(sample_name=sample_name)
                                                               if self.naming_format is not None else None,
                                                               plot_dist=self.plot_dist,
                                                               use_first_guess=self.use_first_guess)

        self._info("\n".join([f"{sample_name}: Expression Threshold: {v[0]}, Non-Expression Threshold: {v[1]}"
                   for sample_name, v in self.output.items()]))

        return self.output[sample_name]