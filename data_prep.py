import numpy as np
import pandas as pd


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    :param filepath: path where data is stored
    :return: pandas dataframe containing prepared data
    """

    df = pd.read_csv(filepath)

    # Some numeric columns should be treated as categorical
    df["Id"] = df.Id.astype(str)
    df["MSSubClass"] = df.MSSubClass.astype(str)

    # null alley is not informative. Replace with NA string to make no alley a category in itself
    df["Alley"] = df.Alley.replace(np.nan, "NA")

    # compute total number of baths
    df['TotalBath'] = df.BsmtFullBath + df.BsmtHalfBath + df.FullBath + df.HalfBath

    # if YearRemodAdd is same as YearBuilt, there was no remodel
    df['YearRemodAdd'] = np.where(
        df.YearRemodAdd == df.YearBuilt, np.nan, df.YearRemodAdd
    )

    return df
