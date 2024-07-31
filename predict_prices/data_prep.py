import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    :param filepath: path where data is stored
    :return: pandas dataframe containing prepared data
    """

    df = pd.read_csv(filepath)

    # Some numeric columns should be treated as categorical
    df["MSSubClass"] = df.MSSubClass.astype(str)

    # Handle nulls in categorical columns by replacing null with Non string.
    # Also make these columns categorical rather than strings
    columns = df.select_dtypes(include='object').columns
    df[columns] = df[columns].apply(
        lambda col: pd.Categorical(np.where(
            col.isna(), 'None', col
        ))
    )

    # ID is unique for each row so it should not be categorical
    df["Id"] = df.Id.astype(str)

    # null alley is not informative. Replace with NA string to make no alley a category in itself
    df["Alley"] = df.Alley.replace(np.nan, "NA")

    # compute total number of baths
    df['TotalBath'] = df.BsmtFullBath + df.BsmtHalfBath + df.FullBath + df.HalfBath

    # if YearRemodAdd is same as YearBuilt, there was no remodel
    df['YearRemodAdd'] = np.where(
        df.YearRemodAdd == df.YearBuilt, 'None', df.YearRemodAdd
    )

    return df


def clean_after_eda(df: pd.DataFrame) -> (pd.DataFrame, IterativeImputer):
    """
    Set ID column to data index, and impute nulls with iterative imputer. Iterative imputer is experimental, and
    documentation can be found at https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
    :param df: pandas dataframe with ID column
    :return:df: cleaned dataframe
    :return:imputer: fit imputer
    """

    df = df.set_index('Id')

    # Iterative imputer only runs on numeric columns, so we need to separate the numeric columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df_numeric = df.select_dtypes(include=numerics)
    df_obj = df.select_dtypes(exclude=numerics)

    imputer = IterativeImputer(max_iter=10, random_state=0)

    df_numeric_impute = imputer.fit_transform(df_numeric)
    df_numeric_impute = pd.DataFrame(df_numeric_impute, columns=df_numeric.columns, index=df_numeric.index)

    # add [cols] to end of this line to return columns in original order
    cols = df.columns
    df = pd.merge(df_obj, df_numeric_impute, left_index=True, right_index=True)[cols]

    return df, imputer


def split_x_y(df: pd.DataFrame, tgt: str, include_categoricals: bool = True, drop: list = []) -> (pd.DataFrame, pd.Series):
    """
    Split data frame into explanatory variables and target variables
    :param df: data frame containing data to be modeled
    :param tgt: target variable
    :param include_categoricals: boolean indicating whether categoricals should be included in the dataframe
    :param drop: extra columns to drop from dataset
    :return: X: dataframe containing x variables, y: dataframe containing target
    """

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    X = df.drop([tgt] + drop, axis=1)
    if not include_categoricals:
        X = X.select_dtypes(include=numerics)
    y = df[tgt]

    return X, y
