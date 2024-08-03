import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
import sklearn.impute as impute
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

_LOT_SHAPE = ['Reg', 'IR1', 'IR2', 'IR3', 'NA']
_UTILITIES = ['AllPub', 'NoSewr', 'NoSeWa', 'ELO', 'NA']
_LAND_SLOPE = ['Gtl', 'Mod', 'Sev', 'NA']
_EXTER_QUAL = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
_EXTER_CON = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
_BSMT_QUAL = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
_BSMT_COND = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
_BSMT_EXPOSURE = ['Gd', 'Av', 'Mn', 'No', 'NA']
_BSMT_FIN_TYPE_1 = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']
_BSMT_FIN_TYPE_2 = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']
_HEATING_QC = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
_CENTRAL_AIR = ['N', 'Y', 'NA']
_KITCHEN_QUAL = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
_FUNCTIONAL = ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal', 'NA']
_FIREPLACE_QU = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
_GARAGE_FINISH = ['Fin', 'RFn', 'Unf', 'NA']
_GARAGE_QUAL = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
_GARAGE_COND = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
_PAVED_DRIVE = ['Y', 'P', 'N', 'NA']
_POOL_QC = ['Ex', 'Gd', 'TA', 'Fa', 'NA']
_FENCE = ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA']

_ORDERED_CATEGORICALS = {
    'LotShape': _LOT_SHAPE,
    'Utilities': _UTILITIES,
    'LandSlope': _LAND_SLOPE,
    'ExterQual': _EXTER_QUAL,
    'ExterCond': _EXTER_CON,
    'BsmtQual': _BSMT_QUAL,
    'BsmtCond': _BSMT_COND,
    'BsmtExposure': _BSMT_EXPOSURE,
    'BsmtFinType1': _BSMT_FIN_TYPE_1,
    'BsmtFinType2': _BSMT_FIN_TYPE_2,
    'CentralAir': _CENTRAL_AIR,
    'KitchenQual': _KITCHEN_QUAL,
    'Functional': _FUNCTIONAL,
    'FireplaceQu': _FIREPLACE_QU,
    'GarageFinish': _GARAGE_FINISH,
    'GarageQual': _GARAGE_QUAL,
    'GarageCond': _GARAGE_COND,
    'PavedDrive': _PAVED_DRIVE,
    'PoolQC': _POOL_QC,
    'Fence': _FENCE
}


def load_data(filepath: str) -> pd.DataFrame:
    """
    read csv into dataframe
    :param filepath: path to csv that should be loaded
    :return: df
    """

    return pd.read_csv(filepath)


def eda_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
        Clean data frame to prepare for eda
        :param df: raw dataframe
        :return: df: cleaned dataframe
        """
    # Some numeric columns should be treated as categorical
    df["MSSubClass"] = df.MSSubClass.astype(str)
    df = df.drop('Utilities', axis=1)

    # Handle nulls in categorical columns by replacing null with Non string.
    # Also make these columns categorical rather than strings so that we can order the ordinal categoricals
    objects = df.select_dtypes(include='object').columns
    df[objects] = df[objects].apply(
        lambda col: pd.Categorical(convert_na_to_string(col), categories=_ORDERED_CATEGORICALS[col.name],
                                   ordered=True) if col.name in _ORDERED_CATEGORICALS else pd.Categorical(convert_na_to_string(col))
    )

    # ID is unique for each row so it should not be categorical
    df["Id"] = df.Id.astype(str)

    # null alley is not informative. Replace with NA string to make no alley a category in itself
    df["Alley"] = df.Alley.replace(np.nan, "NA")

    # compute total number of baths
    df['TotalBath'] = df.BsmtFullBath + df.BsmtHalfBath + df.FullBath + df.HalfBath

    # if YearRemodAdd is same as YearBuilt, there was no remodel
    df['YearRemodAdd'] = np.where(
        df.YearRemodAdd == df.YearBuilt, np.nan, df.YearRemodAdd
    )

    return df


def convert_na_to_string(col):
    return np.where(col.isna(), 'NA', col)


def clean_after_eda(df: pd.DataFrame) -> (pd.DataFrame, impute.IterativeImputer):
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

    imputer = impute.IterativeImputer(max_iter=10, random_state=0)

    df_numeric_impute = imputer.fit_transform(df_numeric)
    df_numeric_impute = pd.DataFrame(df_numeric_impute, columns=df_numeric.columns, index=df_numeric.index)

    # add [cols] to end of this line to return columns in original order
    cols = df.columns
    df = pd.merge(df_obj, df_numeric_impute, left_index=True, right_index=True)[cols]

    return df, imputer


def ordinal_encode(df: pd.DataFrame) -> OrdinalEncoder:
    """
    ordinal encoder for ordinal categorical variables
    :param df:
    :return: dataframe with ordinal encoding of ordinal categorical variables
    """
    enc = OrdinalEncoder()

    ordinals = list(_ORDERED_CATEGORICALS.keys())

    enc.fit(df[ordinals])

    return enc


def ordinal_transform(df: pd.DataFrame, enc: OrdinalEncoder):
    ordinals = list(_ORDERED_CATEGORICALS.keys())
    print(ordinals[1])
    df[ordinals] = enc.transform(df[ordinals])

    return df


def categorical_encoder(df: pd.DataFrame, ohe: bool) -> (pd.DataFrame, OneHotEncoder | OrdinalEncoder):
    """
    Categorical encoding for non-ordinal categorical variables
    :param df:
    :param ohe: boolean, if true use one hot encoder, if false use ordinal encoder
    :return: df
    """
    categorical_columns = df.select_dtypes(include='category').columns
    if ohe:
        cat_enc = OneHotEncoder(sparse_output=False)
        cat_enc.fit(df[categorical_columns])
        one_hot_encoded = cat_enc.transform(df[categorical_columns])

        one_hot_df = pd.DataFrame(one_hot_encoded, columns=cat_enc.get_feature_names_out(categorical_columns),
                                  index=df.index)
        df = pd.merge(df, one_hot_df, left_index=True, right_index=True)

        # Drop the original categorical columns
        df = df.drop(categorical_columns, axis=1)
    else:
        cat_enc = OrdinalEncoder()
        cat_enc.fit(df[categorical_columns])
        df[categorical_columns] = cat_enc.transform(df[categorical_columns])

    return df, cat_enc


def split_x_y(df: pd.DataFrame, tgt: str, include_categoricals: bool = True, drop: list = []) -> (
        pd.DataFrame, pd.Series):
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
