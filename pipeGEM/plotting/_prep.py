import pandas as pd


def prep_flux_df(flux_df, rxn_ids) -> (pd.DataFrame, pd.DataFrame):
    model_df = flux_df.loc[:, [i for i in flux_df.columns if i not in rxn_ids]]
    flux_df = flux_df.copy().loc[:, rxn_ids if isinstance(rxn_ids, list) else list(rxn_ids.keys())]
    if isinstance(rxn_ids, dict):
        flux_df = flux_df.rename(columns=rxn_ids)
    return model_df, flux_df


def prep_fva_plotting_data(min_df, max_df):
    min_df = min_df.copy()
    max_df = max_df.copy()
    if not min_df.shape == max_df.shape:
        raise ValueError("Max df should have the same shape with min df")

    median = (min_df + max_df) / 2
    upper = (max_df + max_df - median)
    lower = (min_df + min_df - median)
    for df in [min_df, max_df, median, upper, lower]:
        df["model"] = df.index
    mg = pd.concat([min_df, max_df, median, upper, lower], axis=0, ignore_index=True)
    return mg.melt(id_vars=["model"], var_name="Reactions", value_name="Flux")