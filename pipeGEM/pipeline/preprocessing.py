from typing import List, Dict, Any, Optional

import cobra
import pandas as pd

from ._base import Pipeline
from pipeGEM.data.preprocessing import (get_gene_id_map, translate_gene_id, unify_score_column,
                                        transform_HPA_data, get_discretize_data)
from pipeGEM.data.fetching import fetch_HPA_data, load_HPA_data


# data converter, loader, transformer,
class GeneDataIDConverter(Pipeline):
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
            ) -> pd.DataFrame:
        self.output = translate_gene_id(data_df=data_df,
                                        map_df=map_df,
                                        gene_col=gene_col,
                                        to_id=to_id)
        return self.output["data_df"]


class GeneDataDiscretizer(Pipeline):
    def __init__(self,
                 data_df):
        super().__init__()
        self.data_df = data_df

    def run(self,
            sample_names,
            expr_threshold_dic,
            non_expr_threshold_dic
            ) -> pd.DataFrame:
        output = get_discretize_data(sample_names=sample_names,
                                     data_df=self.data_df,
                                     expr_threshold_dic=expr_threshold_dic,
                                     non_expr_threshold_dic=non_expr_threshold_dic)
        return output["data_df"]


class ProteinDataLoader(Pipeline):
    def __init__(self,
                 data_name,
                 data_path):
        super().__init__()
        self.data_df = pd.read_csv(fetch_HPA_data(data_name,
                                                  data_path)["data_path"])

    def run(self,
            cancer_query="all",
            tissue_query="all",
            cell_type_query="all",
            cluster_query="all",
            reliability_query="default") -> pd.DataFrame:
        if reliability_query == "default":
            reliability_query = ["Enhanced", "Approved", "Supported"]

        df_query_kw = {"Cancer": cancer_query,
                       "Tissue": tissue_query,
                       "Cell type": cell_type_query,
                       "Cluster": cluster_query,
                       "Reliability": reliability_query}

        data_df = self.data_df.copy()
        for c in self.data_df.columns:
            if c in df_query_kw and df_query_kw[c] != "all":
                data_df = data_df.loc[data_df[c].isin(df_query_kw[c]), :]
        self.output = data_df
        return self.output


class ProteinDataTransformer(Pipeline):
    def __init__(self,
                 level_dic=None,
                 categories=None,
                 score_col_name="score"):
        super().__init__()

        self.level_dic = {"High": 3,
                          "Medium": 2,
                          "Low": 1,
                          "Not detected": -1}
        if level_dic is not None:
            self.level_dic.update(level_dic)
        self.score_col_name = score_col_name
        self.categories = ["Tissue"] if categories is None else categories

    def run(self,
            data_df: pd.DataFrame) -> pd.DataFrame:
        u_output = unify_score_column(data_df=data_df,
                                      level_dic=self.level_dic,
                                      score_col_name=self.score_col_name)
        output = transform_HPA_data(data_df=u_output["data_df"],
                                    categories=self.categories,
                                    score_col_name=self.score_col_name)
        self.output = output
        self.output["used_rxn_thres"] = u_output["used_rxn_thres"]
        return output
