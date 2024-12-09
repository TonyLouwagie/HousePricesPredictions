from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn import impute, preprocessing  # type: ignore

from data_load import RawTestingData, RawTrainingData

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


@dataclass
class TrainData:
    X: pd.DataFrame
    y: pd.Series


@dataclass
class CategoricalEncoders:
    # TODO: can ordinal_encoder and categorical_encoder be combined?
    ordinal_encoder: preprocessing.OrdinalEncoder
    categorical_encoder: preprocessing.OneHotEncoder | preprocessing.OrdinalEncoder


@dataclass
class TrainDataPrepOutputs:
    train_data: TrainData
    categorical_encoders: CategoricalEncoders


@dataclass
class TrainDataPrepInputs:
    train_data: RawTrainingData
    ohe_bool: bool
    target_variable: str

    def train_data_prep(self) -> TrainDataPrepOutputs:
        train_df = _eda_clean(self.train_data.df)
        train_df = _clean_after_eda(train_df)

        ord_enc = _ordinal_encode(train_df)
        train_df = _ordinal_transform(train_df, ord_enc)

        cat_enc = _categorical_encoder(train_df, self.ohe_bool)
        train_df = _categorical_transform(train_df, cat_enc)

        train_data = _split_x_y(train_df, self.target_variable)
        categorical_encoders = CategoricalEncoders(ord_enc, cat_enc)

        return TrainDataPrepOutputs(train_data, categorical_encoders)


@dataclass
class TestDataPrepInputs:
    test_df: RawTestingData
    categorical_encoders: CategoricalEncoders

    def test_data_prep(self) -> pd.DataFrame:
        test_df = _eda_clean(self.test_df.df)
        test_df = _clean_after_eda(test_df)

        test_df = _ordinal_transform(test_df, self.categorical_encoders.ordinal_encoder)

        test_X = _categorical_transform(test_df, self.categorical_encoders.categorical_encoder)

        return test_X


def _eda_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data frame to prepare for eda
    """
    # Some numeric columns should be treated as categorical
    df["MSSubClass"] = df.MSSubClass.astype(str)

    # Handle nulls in categorical columns by replacing null with Non string.
    # Also make these columns categorical rather than strings so that we can order the ordinal categoricals
    objects = df.select_dtypes(include=object).columns
    df[objects] = df[objects].apply(
        lambda col: pd.Categorical(
            _convert_na_to_string(col),
            categories=_ORDERED_CATEGORICALS[col.name],
            ordered=True
        ) if col.name in _ORDERED_CATEGORICALS else pd.Categorical(_convert_na_to_string(col)))

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


def _convert_na_to_string(col):
    return np.where(col.isna(), 'NA', col)


def _clean_after_eda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set ID column to data index, and impute nulls with iterative imputer. Iterative imputer is experimental, and
    documentation can be found at https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
    :param df: pandas dataframe with ID column
    :return:tuple[df, imputer]: cleaned dataframe
    """

    df = df.set_index('Id')

    # Iterative imputer only runs on numeric columns, so we need to separate the numeric columns
    df_numeric = df.select_dtypes(include=[int, float])
    df_obj = df.select_dtypes(exclude=[int, float])

    imputer = impute.KNNImputer()

    df_numeric_impute = imputer.fit_transform(df_numeric)
    df_numeric_impute = pd.DataFrame(df_numeric_impute, columns=df_numeric.columns, index=df_numeric.index)

    # add [cols] to end of this line to return columns in original order
    cols = df.columns
    df = pd.merge(df_obj, df_numeric_impute, left_index=True, right_index=True)[cols]

    return df


def _ordinal_encode(df: pd.DataFrame) -> preprocessing.OrdinalEncoder:
    """
    fit ordinal encoder for ordinal categorical variables
    """
    enc = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    ordinals = list(_ORDERED_CATEGORICALS.keys())

    enc.fit(df[ordinals])

    return enc


def _ordinal_transform(df: pd.DataFrame, enc: preprocessing.OrdinalEncoder) -> pd.DataFrame:
    """
    use fit ordinal encoder to transform ordinal columns
    """
    ordinals = list(_ORDERED_CATEGORICALS.keys())
    df[ordinals] = enc.transform(df[ordinals])

    return df


# TODO: Create an enum class that is OneHotEncoder or OrdinalEncoder only. Return that class here.
def _categorical_encoder(df: pd.DataFrame, ohe: bool) -> (preprocessing.OneHotEncoder | preprocessing.OrdinalEncoder):
    """
    Categorical encoding for non-ordinal categorical variables
    """
    categorical_columns = df.select_dtypes(include='category').columns
    if ohe:
        cat_enc = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_enc.fit(df[categorical_columns])
    else:
        cat_enc = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        cat_enc.fit(df[categorical_columns])

    return cat_enc


def _categorical_transform(
        df: pd.DataFrame,
        cat_enc: preprocessing.OrdinalEncoder | preprocessing.OneHotEncoder
) -> pd.DataFrame:
    """
    Use fit categorical encoder to transform dataset
    """
    categorical_columns = df.select_dtypes(include='category').columns

    if isinstance(cat_enc, preprocessing.OneHotEncoder):
        one_hot_encoded = cat_enc.transform(df[categorical_columns])
        one_hot_df = pd.DataFrame(one_hot_encoded, columns=cat_enc.get_feature_names_out(categorical_columns),
                                  index=df.index)
        df = pd.merge(df, one_hot_df, left_index=True, right_index=True)
        # Drop the original categorical columns
        df = df.drop(categorical_columns, axis=1)
    else:
        df[categorical_columns] = cat_enc.transform(df[categorical_columns])

    return df


def _split_x_y(df: pd.DataFrame, target_variable: str, drop=None) -> TrainData:
    """
    Split data frame into explanatory variables and target variables
    """

    if drop is None:
        drop = []
    X = df.drop([target_variable] + drop, axis=1)
    # The metric of interest is log transformed RMSE
    y = np.log(df[target_variable])

    return TrainData(X, y)
