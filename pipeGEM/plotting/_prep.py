import pandas as pd


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