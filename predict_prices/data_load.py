from dataclasses import dataclass

import pandas as pd
import pandera as pa

_DTYPE_DICT = {
    "Id": int,
    "MSSubClass": int,
    "MSZoning": str,
    "LotFrontage": "Int64",
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
    "MasVnrArea": "Int64",
    "ExterQual": str,
    "ExterCond": str,
    "Foundation": str,
    "BsmtQual": str,
    "BsmtCond": str,
    "BsmtExposure": str,
    "BsmtFinType1": str,
    "BsmtFinSF1": "Int64",
    "BsmtFinType2": str,
    "BsmtFinSF2": "Int64",
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
    "GarageYrBlt": "Int64",
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
    "LotFrontage": pa.Column(int, nullable=True),
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
    "MasVnrArea": pa.Column(int, nullable=True),
    "ExterQual": pa.Column(str),
    "ExterCond": pa.Column(str),
    "Foundation": pa.Column(str),
    "BsmtQual": pa.Column(str, nullable=True),
    "BsmtCond": pa.Column(str, nullable=True),
    "BsmtExposure": pa.Column(str, nullable=True),
    "BsmtFinType1": pa.Column(str, nullable=True),
    "BsmtFinSF1": pa.Column(int, nullable=True),
    "BsmtFinType2": pa.Column(str, nullable=True),
    "BsmtFinSF2": pa.Column(int, nullable=True),
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
    "GarageYrBlt": pa.Column(int, nullable=True),
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
