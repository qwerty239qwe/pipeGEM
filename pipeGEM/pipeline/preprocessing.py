from typing import List, Dict, Any, Optional

import cobra
import pandas as pd

from ._job import Pipeline
from pipeGEM.data.preprocessing import get_gene_id_map, translate_gene_id


class GeneIDMapper(Pipeline):
    def __init__(self,
                 gene_names: List[str],
                 from_id: str,
                 to_id: str,
                 df_path: str,
                 ds_kws: Dict[str, Any],
                 map_type: str = "df",
                 drop_unused: bool = False,
                 ref_model: Optional[cobra.Model] = None
                 ):
        super().__init__()
        self.map_df = get_gene_id_map(gene_names=gene_names,
                                      from_id=from_id,
                                      to_id=to_id,
                                      df_path=df_path,
                                      ds_kws=ds_kws,
                                      map_type=map_type,
                                      drop_unused=drop_unused,
                                      ref_model=ref_model)["map_df"]

    def run(self,
            data_df: pd.DataFrame,
            map_df: pd.DataFrame,
            gene_col: str,
            to_id: str
            ):
        self.output = translate_gene_id(data_df=data_df,
                                        map_df=map_df,
                                        gene_col=gene_col,
                                        to_id=to_id)
        return self.output["data_df"]