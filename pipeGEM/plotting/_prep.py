import pandas as pd


def prep_flux_df(flux_df: pd.DataFrame,
                 rxn_ids) -> (pd.DataFrame, pd.DataFrame):
    model_df = flux_df.loc[:, [i for i in flux_df.columns if i not in rxn_ids]]
    flux_df = flux_df.copy().reindex(columns= rxn_ids if isinstance(rxn_ids, list) else list(rxn_ids.keys()))
    if isinstance(rxn_ids, dict):
        flux_df = flux_df.rename(columns=rxn_ids)
    return model_df, flux_df


def prep_fva_plotting_data(fva_df):
    fva_df = fva_df.copy()
    fva_df["median"] = (fva_df["minimum"] + fva_df["maximum"]) / 2
    fva_df["upper"] = (fva_df["maximum"] + fva_df["maximum"] - fva_df["median"])
    fva_df["lower"] = (fva_df["minimum"] + fva_df["minimum"] - fva_df["median"])
    return fva_df.melt(id_vars=["name", "Reaction"], var_name="Stats", value_name="Flux")


def filter_fva_df(fva_df: pd.DataFrame, threshold, verbosity):
    diff = abs(fva_df["maximum"] - fva_df["minimum"]) < threshold
    if verbosity > 0:
        filtered = diff.all(axis=0)
        print("Filtering out rxns (abs(max - min) < theshold): ", filtered[filtered].index.to_list())
    return fva_df.loc[:, (~diff).any(axis=0)]