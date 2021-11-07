import pandas as pd


def prep_fva_plotting_data(min_df, max_df):
    if not min_df.shape == max_df.shape:
        raise ValueError("Max df should have the same shape with min df")

    median = (min_df.iloc[0, :] + max_df.iloc[1, :]) / 2
    upper = (max_df + max_df - median)
    lower = (min_df - min_df + median)
    return pd.concat([min_df, max_df, median, upper, lower], axis=0)