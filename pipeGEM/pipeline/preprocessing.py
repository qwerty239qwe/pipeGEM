from typing import List, Dict, Any, Optional
from functools import partial

import cobra
import pandas as pd
import numpy as np

from ._base import Pipeline
from pipeGEM.data.preprocessing import (get_gene_id_map, translate_gene_id, unify_score_column,
                                        transform_HPA_data, )

from pipeGEM.data.fetching import fetch_HPA_data, load_HPA_data
from pipeGEM.integration.mapping import Expression


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
                 ref_model: Optional[cobra.Model] = None,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
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


class GeneDataLengthGetter(Pipeline):
    def __init__(self,
                 gene_names: List[str],
                 from_id: str,
                 df_path: str,
                 ds_kws: Dict[str, Any],
                 map_type: str = "df",
                 drop_unused: bool = False,
                 ref_model: Optional[cobra.Model] = None,
                 *args, **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.map_df = get_gene_id_map(gene_names=gene_names,
                                      from_id=from_id,
                                      to_id="transcript_length",
                                      df_path=df_path,
                                      ds_kws=ds_kws,
                                      map_type=map_type,
                                      drop_unused=drop_unused,
                                      ref_model=ref_model)["map_df"]

    def run(self,
            data_df: pd.DataFrame,
            gene_col: str
            ) -> pd.DataFrame:
        self.output = translate_gene_id(data_df=data_df,
                                        map_df=self.map_df,
                                        gene_col=gene_col,
                                        to_id="transcript_length")
        return self.output["data_df"]


class GeneDataDiscretizer(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def run(self,
            data_df,
            sample_names,
            expr_threshold_dic,
            non_expr_threshold_dic,
            use_interp=False,
            ) -> pd.DataFrame:
        from pipeGEM.integration.utils import get_discretize_data, get_interpolate_data

        self.output = get_discretize_data(sample_names=sample_names,
                                          data_df=data_df,
                                          expr_threshold_dic=expr_threshold_dic,
                                          non_expr_threshold_dic=non_expr_threshold_dic) if not use_interp else \
                      get_interpolate_data(sample_names=sample_names,
                                           data_df=data_df,
                                           expr_threshold_dic=expr_threshold_dic,
                                           non_expr_threshold_dic=non_expr_threshold_dic)
        return self.output


class GeneDataLinearScaler(Pipeline):
    def __init__(self):
        super().__init__()

    def run(self,
            data,
            domain_lb,
            domain_ub,
            range_lb=0,
            range_ub=1,
            range_nan=0.5,
            *args,
            **kwargs):
        ls_func = partial(np.interp, xp=[domain_lb, domain_ub], fp=[range_lb, range_ub])
        self.output = {name: ls_func(val) if not np.isnan(val) else range_nan
                       for name, val in data.items()}
        return self.output


class GeneData(Pipeline):
    def __init__(self, *args, **kwargs):
        super(GeneData, self).__init__(*args, **kwargs)
        self._expression = None

    def run(self, data, model, *args, **kwargs):
        self._expression = Expression(model, data, *args, **kwargs)
        self.output = self._expression
        return self.output


class GeneDataSet(Pipeline):
    def __init__(self, data_df: pd.DataFrame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_df = data_df
        self.jobs = {c: GeneData() for c in data_df.columns}
        self._expression_dict = {}

    def __getitem__(self, item):
        return self._expression_dict[item]

    def items(self):
        return self._expression_dict.items()

    def run(self, model, *args, **kwargs):
        for c, job in self.jobs.items():
            self._expression_dict[c] = job(data=self.data_df[c],
                                           model=model[c] if isinstance(model, dict) else model,
                                           *args, **kwargs)
        self.output = self._expression_dict
        return self.output


class ProteinDataLoader(Pipeline):
    def __init__(self,
                 data_name,
                 data_path):
        super().__init__()
        self.data_name = data_name
        self.data_path = data_path

    def run(self,
            cancer_query="all",
            tissue_query="all",
            cell_type_query="all",
            cluster_query="all",
            reliability_query="default") -> pd.DataFrame:
        self.data_df = pd.read_csv(fetch_HPA_data(self.data_name,
                                                  self.data_path)["data_path"])

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


class HPADataFetcher(Pipeline):
    def __init__(self,
                 data_name,
                 data_path,
                 level_dic=None,
                 categories=None,
                 score_col_name="score",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ptn_loader = ProteinDataLoader(data_name=data_name,
                                            data_path=data_path)
        self.ptn_transformer = ProteinDataTransformer(level_dic=level_dic,
                                                      categories=categories,
                                                      score_col_name=score_col_name)

    def run(self,
            cancer_query="all",
            tissue_query="all",
            cell_type_query="all",
            cluster_query="all",
            reliability_query="default",
            *args, **kwargs):
        loaded = self.ptn_loader(cancer_query=cancer_query,
                                 tissue_query=tissue_query,
                                 cell_type_query=cell_type_query,
                                 cluster_query=cluster_query,
                                 reliability_query=reliability_query)
        self.output = self.ptn_transformer(loaded)
        return self.output