from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandera as pa
from sklearn import impute, preprocessing  # type: ignore

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

_DTYPE_DICT = {
    "Id": int,
    "MSSubClass": int,
    "MSZoning": str,
    "LotFrontage": float,
    "LotArea": int,
    "Street": str,
    "Alley": str,
    "LotShape": str,
    "LandContour": str,
    "Utilities": str,
    "LotConfig": str,
    "LandSlope": str,
    "Neighborhood": str,
    "Condition1": str,
    "Condition2": str,
    "BldgType": str,
    "HouseStyle": str,
    "OverallQual": int,
    "OverallCond": int,
    "YearBuilt": int,
    "YearRemodAdd": int,
    "RoofStyle": str,
    "RoofMatl": str,
    "Exterior1st": str,
    "Exterior2nd": str,
    "MasVnrType": str,
    "MasVnrArea": float,
    "ExterQual": str,
    "ExterCond": str,
    "Foundation": str,
    "BsmtQual": str,
    "BsmtCond": str,
    "BsmtExposure": str,
    "BsmtFinType1": str,
    "BsmtFinSF1": float,
    "BsmtFinType2": str,
    "BsmtFinSF2": float,
    "BsmtUnfSF": "Int64",
    "TotalBsmtSF": "Int64",
    "Heating": str,
    "HeatingQC": str,
    "CentralAir": str,
    "Electrical": str,
    "1stFlrSF": int,
    "2ndFlrSF": int,
    "LowQualFinSF": int,
    "GrLivArea": int,
    "BsmtFullBath": "Int64",
    "BsmtHalfBath": "Int64",
    "FullBath": int,
    "HalfBath": int,
    "BedroomAbvGr": int,
    "KitchenAbvGr": int,
    "KitchenQual": str,
    "TotRmsAbvGrd": int,
    "Functional": str,
    "Fireplaces": int,
    "FireplaceQu": str,
    "GarageType": str,
    "GarageYrBlt": float,
    "GarageFinish": str,
    "GarageCars": "Int64",
    "GarageArea": "Int64",
    "GarageQual": str,
    "GarageCond": str,
    "PavedDrive": str,
    "WoodDeckSF": int,
    "OpenPorchSF": int,
    "EnclosedPorch": int,
    "3SsnPorch": int,
    "ScreenPorch": int,
    "PoolArea": int,
    "PoolQC": str,
    "Fence": str,
    "MiscFeature": str,
    "MiscVal": int,
    "MoSold": int,
    "YrSold": int,
    "SaleType": str,
    "SaleCondition": str,
}

_TARGET_DTYPE = {"SalePrice": int}

_RAW_FEATURES = {
    "Id": pa.Column(int),
    "MSSubClass": pa.Column(int),
    "MSZoning": pa.Column(str, nullable=True),
    "LotFrontage": pa.Column(float, nullable=True),
    "LotArea": pa.Column(int),
    "Street": pa.Column(str),
    "Alley": pa.Column(str, nullable=True),
    "LotShape": pa.Column(str),
    "LandContour": pa.Column(str),
    "Utilities": pa.Column(str, nullable=True),
    "LotConfig": pa.Column(str),
    "LandSlope": pa.Column(str),
    "Neighborhood": pa.Column(str),
    "Condition1": pa.Column(str),
    "Condition2": pa.Column(str),
    "BldgType": pa.Column(str),
    "HouseStyle": pa.Column(str),
    "OverallQual": pa.Column(int),
    "OverallCond": pa.Column(int),
    "YearBuilt": pa.Column(int),
    "YearRemodAdd": pa.Column(int),
    "RoofStyle": pa.Column(str),
    "RoofMatl": pa.Column(str),
    "Exterior1st": pa.Column(str, nullable=True),
    "Exterior2nd": pa.Column(str, nullable=True),
    "MasVnrType": pa.Column(str, nullable=True),
    "MasVnrArea": pa.Column(float, nullable=True),
    "ExterQual": pa.Column(str),
    "ExterCond": pa.Column(str),
    "Foundation": pa.Column(str),
    "BsmtQual": pa.Column(str, nullable=True),
    "BsmtCond": pa.Column(str, nullable=True),
    "BsmtExposure": pa.Column(str, nullable=True),
    "BsmtFinType1": pa.Column(str, nullable=True),
    "BsmtFinSF1": pa.Column(float, nullable=True),
    "BsmtFinType2": pa.Column(str, nullable=True),
    "BsmtFinSF2": pa.Column(float, nullable=True),
    "BsmtUnfSF": pa.Column(int, nullable=True),
    "TotalBsmtSF": pa.Column(int, nullable=True),
    "Heating": pa.Column(str),
    "HeatingQC": pa.Column(str),
    "CentralAir": pa.Column(str),
    "Electrical": pa.Column(str, nullable=True),
    "1stFlrSF": pa.Column(int),
    "2ndFlrSF": pa.Column(int),
    "LowQualFinSF": pa.Column(int),
    "GrLivArea": pa.Column(int),
    "BsmtFullBath": pa.Column(int, nullable=True),
    "BsmtHalfBath": pa.Column(int, nullable=True),
    "FullBath": pa.Column(int),
    "HalfBath": pa.Column(int),
    "BedroomAbvGr": pa.Column(int),
    "KitchenAbvGr": pa.Column(int),
    "KitchenQual": pa.Column(str, nullable=True),
    "TotRmsAbvGrd": pa.Column(int),
    "Functional": pa.Column(str, nullable=True),
    "Fireplaces": pa.Column(int),
    "FireplaceQu": pa.Column(str, nullable=True),
    "GarageType": pa.Column(str, nullable=True),
    "GarageYrBlt": pa.Column(float, nullable=True),
    "GarageFinish": pa.Column(str, nullable=True),
    "GarageCars": pa.Column(int, nullable=True),
    "GarageArea": pa.Column(int, nullable=True),
    "GarageQual": pa.Column(str, nullable=True),
    "GarageCond": pa.Column(str, nullable=True),
    "PavedDrive": pa.Column(str),
    "WoodDeckSF": pa.Column(int),
    "OpenPorchSF": pa.Column(int),
    "EnclosedPorch": pa.Column(int),
    "3SsnPorch": pa.Column(int),
    "ScreenPorch": pa.Column(int),
    "PoolArea": pa.Column(int),
    "PoolQC": pa.Column(str, nullable=True),
    "Fence": pa.Column(str, nullable=True),
    "MiscFeature": pa.Column(str, nullable=True),
    "MiscVal": pa.Column(int),
    "MoSold": pa.Column(int),
    "YrSold": pa.Column(int),
    "SaleType": pa.Column(str, nullable=True),
    "SaleCondition": pa.Column(str),
}

_TARGET = {"SalePrice": pa.Column(int)}


@dataclass
class RawTrainData:
    df: pd.DataFrame
    schema = pa.DataFrameSchema(_RAW_FEATURES | _TARGET)

    def __post_init__(self):
        self.schema.validate(self.df)


@dataclass
class RawTestingData:
    df: pd.DataFrame
    schema = pa.DataFrameSchema(_RAW_FEATURES)

    def __post_init__(self):
        self.schema.validate(self.df)


def load_training_data(filepath: str) -> RawTrainData:
    """
    read csv into dataframe
    :param filepath: path to csv that should be loaded
    :return: df
    """

    return RawTrainData(pd.read_csv(filepath, dtype= _DTYPE_DICT|_TARGET_DTYPE))


def load_testing_data(filepath: str) -> RawTestingData:
    """
    read csv into dataframe
    :param filepath: path to csv that should be loaded
    :return: df
    """

    return RawTestingData(pd.read_csv(filepath, dtype= _DTYPE_DICT))


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
    train_data: RawTrainData
    ohe_bool: bool
    target_variable: str

    def train_data_prep(self) -> TrainDataPrepOutputs:
        train_df = _eda_clean(self.train_data.df)
        train_df, _ = _clean_after_eda(train_df)

        ord_enc = _ordinal_encode(train_df)
        train_df = _ordinal_transform(train_df, ord_enc)

        cat_enc = _categorical_encoder(train_df, self.ohe_bool)
        train_df = _categorical_transform(train_df, cat_enc)

        train_data = _split_x_y(train_df, self.target_variable)
        categorical_encoders = CategoricalEncoders(ord_enc, cat_enc)

        return TrainDataPrepOutputs(train_data, categorical_encoders)


@dataclass
class TestDataPrepInputs:
    test_df: pd.DataFrame
    categorical_encoders: CategoricalEncoders

    def test_data_prep(self) -> pd.DataFrame:
        test_df = _eda_clean(self.test_df)
        test_df, _ = _clean_after_eda(test_df)

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


def _clean_after_eda(df: pd.DataFrame) -> tuple[pd.DataFrame, impute.KNNImputer]:
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

    return df, imputer


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


def _categorical_transform(df: pd.DataFrame,
                           cat_enc: preprocessing.OrdinalEncoder | preprocessing.OneHotEncoder) -> pd.DataFrame:
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
