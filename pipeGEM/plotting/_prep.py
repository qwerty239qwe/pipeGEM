import pandas as pd


def prep_flux_df(flux_df: pd.DataFrame,
                 rxn_ids) -> (pd.DataFrame, pd.DataFrame):
    model_df = flux_df.loc[:, [i for i in flux_df.columns if i not in rxn_ids]]
    flux_df = flux_df.copy().reindex(columns= rxn_ids if isinstance(rxn_ids, list) else list(rxn_ids.keys()))
    if isinstance(rxn_ids, dict):
        flux_df = flux_df.rename(columns=rxn_ids)
    return model_df, flux_df


def prep_fva_plotting_data(min_df,
                           max_df,
                           model_info: pd.Series):
    min_df = min_df.copy()
    max_df = max_df.copy()
    if not min_df.shape == max_df.shape:
        raise ValueError("Max df should have the same shape with min df")

    median = (min_df + max_df) / 2
    upper = (max_df + max_df - median)
    lower = (min_df + min_df - median)
    for df in [min_df, max_df, median, upper, lower]:
        df["model"] = model_info
    mg = pd.concat([min_df, max_df, median, upper, lower], axis=0, ignore_index=True)
    return mg.melt(id_vars=["model"], var_name="Reactions", value_name="Flux")


def filter_fva_df(min_df: pd.DataFrame, max_df: pd.DataFrame, threshold, verbosity):
    diff = abs(max_df - min_df) < threshold
    if verbosity > 0:
        filtered = diff.all(axis=0)
        print("Filtering out rxns (abs(max - min) < theshold): ", filtered[filtered].index.to_list())
    return min_df.loc[:, (~diff).any(axis=0)], max_df.loc[:, (~diff).any(axis=0)]