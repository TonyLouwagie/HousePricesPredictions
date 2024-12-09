from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pandera as pa
from pandera.typing import Series

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

_MSSubClass_Options = [20,30,40,45,50,60,70,75,80,85,90,120,150,160,180,190]
_MSZoning_Options = ['A','C','FV','I','RH','RL','RP','RM','C (all)']
_Street_Options = ['Grvl','Pave']
_Alley_Options = ['Grvl','Pave','NA']
_LotShape_Options = ['Reg','IR1','IR2','IR3']
_LandCountour_Options = ['Lvl','Bnk','HLS','Low']
_Utilities_Options = ['AllPub','NoSewr','NoSeWa','ELO']
_LotConfig_Options = ['Inside','Corner','CulDSac','FR2','FR3']
_LandSlope_Options = ['Gtl','Mod','Sev']
_Neighborhood_Options = ['Blmngtn','Blueste','BrDale','BrkSide','ClearCr','CollgCr','Crawfor','Edwards','Gilbert',
                         'IDOTRR','MeadowV','Mitchel','NAmes','NoRidge','NPkVill','NridgHt','NWAmes','OldTown','SWISU',
                         'Sawyer','SawyerW','Somerst','StoneBr','Timber','Veenker']
_Condition_Options = ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe']
_BldgType_Options = ['1Fam','2fmCon','Duplex','TwnhsE','Twnhs']
_HouseStyle_Options = ['2Story','1Story','1.5Fin','1.5Unf','SFoyer','SLvl','2.5Unf','2.5Fin']
_RoofStyle_Options = ['Flat','Gable','Gambrel','Hip','Mansard','Shed']
_RoofMatl_Options = ['ClyTile','CompShg','Membran','Metal','Roll','Tar&Grv','WdShake','WdShngl']
_Exterior_Options = ['AsbShng', 'AsphShn', 'BrkComm', 'Brk Cmn', 'BrkFace', 'CBlock', 'CmentBd', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd',
                        'Other','Plywood','PreCast','Stone','Stucco','VinylSd','Wd Sdng','Wd Shng', 'WdShing']
_MasVnrTyp_Options = ['BrkCmn','BrkFace','CBlock','None','Stone']
_Quality_Options = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
_Foundation_Options = ['BrkTil','CBlock','PConc','Slab','Stone','Wood']
_QualityNA_Options = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
_BsmtExposure_Options = ['Gd','Av','Mn','No','NA']
_BsmtFinType_Options = ['GLQ','ALQ','BLQ','Rec','LwQ','Unf','NA']
_Heating_Options = ['Floor','GasA','GasW','Grav','OthW','Wall']
_CentralAir_Options = ['N','Y']
_Electrical_Options = ['SBrkr','FuseA','FuseF','FuseP','Mix']
_Functional_Options = ['Typ','Min1','Min2','Mod','Maj1','Maj2','Sev','Sal']
_GarageType_Options = ['2Types','Attchd','Basment','BuiltIn','CarPort','Detchd','NA']
_GarageFinish_Options = ['Fin','RFn','Unf','NA']
_PavedDrive_Options = ['Y','P','N']
_FenceQuality_Options = ['GdPrv','MnPrv','GdWo','MnWw','NA']
_MiscFeature_Options = ['Elev','Gar2','Othr','Shed','TenC','NA']
_SaleType_Options = ['WD','CWD','VWD','New','COD','Con','ConLw','ConLI','ConLD','Oth']
_SaleCondition_Options = ['Normal','Abnorml','AdjLand','Alloca','Family','Partial']

_RAW_FEATURES = {
    "Id": pa.Column(int, pa.Check.greater_than(0)),
    "MSSubClass": pa.Column(int, pa.Check.isin(_MSSubClass_Options)),
    "MSZoning": pa.Column(str, pa.Check.isin(_MSZoning_Options), nullable=True),
    "LotFrontage": pa.Column(int, pa.Check.greater_than(0), nullable=True),
    "LotArea": pa.Column(int, pa.Check.greater_than(0)),
    "Street": pa.Column(str, pa.Check.isin(_Street_Options)),
    "Alley": pa.Column(str, pa.Check.isin(_Alley_Options), nullable=True),
    "LotShape": pa.Column(str, pa.Check.isin(_LotShape_Options)),
    "LandContour": pa.Column(str, pa.Check.isin(_LandCountour_Options)),
    "Utilities": pa.Column(str, pa.Check.isin(_Utilities_Options), nullable=True),
    "LotConfig": pa.Column(str, pa.Check.isin(_LotConfig_Options)),
    "LandSlope": pa.Column(str, pa.Check.isin(_LandSlope_Options)),
    "Neighborhood": pa.Column(str, pa.Check.isin(_Neighborhood_Options)),
    "Condition1": pa.Column(str, pa.Check.isin(_Condition_Options)),
    "Condition2": pa.Column(str, pa.Check.isin(_Condition_Options)),
    "BldgType": pa.Column(str, pa.Check.isin(_BldgType_Options)),
    "HouseStyle": pa.Column(str, pa.Check.isin(_HouseStyle_Options)),
    "OverallQual": pa.Column(int, pa.Check.isin(range(1,11))),
    "OverallCond": pa.Column(int, pa.Check.isin(range(1,11))),
    "YearBuilt": pa.Column(int, pa.Check.greater_than(1799)),
    "YearRemodAdd": pa.Column(int, pa.Check.greater_than(1799)),
    "RoofStyle": pa.Column(str, pa.Check.isin(_RoofStyle_Options)),
    "RoofMatl": pa.Column(str, pa.Check.isin(_RoofMatl_Options)),
    "Exterior1st": pa.Column(str, pa.Check.isin(_Exterior_Options), nullable=True),
    "Exterior2nd": pa.Column(str, pa.Check.isin(_Exterior_Options), nullable=True),
    "MasVnrType": pa.Column(str, pa.Check.isin(_MasVnrTyp_Options), nullable=True),
    "MasVnrArea": pa.Column(int, pa.Check.greater_than_or_equal_to(0), nullable=True),
    "ExterQual": pa.Column(str, pa.Check.isin(_Quality_Options)),
    "ExterCond": pa.Column(str, pa.Check.isin(_Quality_Options)),
    "Foundation": pa.Column(str, pa.Check.isin(_Foundation_Options)),
    "BsmtQual": pa.Column(str, pa.Check.isin(_QualityNA_Options), nullable=True),
    "BsmtCond": pa.Column(str, pa.Check.isin(_QualityNA_Options), nullable=True),
    "BsmtExposure": pa.Column(str, pa.Check.isin(_BsmtExposure_Options), nullable=True),
    "BsmtFinType1": pa.Column(str, pa.Check.isin(_BsmtFinType_Options), nullable=True),
    "BsmtFinSF1": pa.Column(int, pa.Check.greater_than_or_equal_to(0), nullable=True),
    "BsmtFinType2": pa.Column(str, pa.Check.isin(_BsmtFinType_Options), nullable=True),
    "BsmtFinSF2": pa.Column(int, nullable=True),
    "BsmtUnfSF": pa.Column(int, pa.Check.greater_than_or_equal_to(0), nullable=True),
    "TotalBsmtSF": pa.Column(int, pa.Check.greater_than_or_equal_to(0), nullable=True),
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


class RawFeatureSchema(pa.DataFrameModel):
    Id: Series[int] = pa.Field(gt=0)
    MSSubClass: Series[int] = pa.Field(isin=_MSSubClass_Options)
    MSZoning: Series[str] = pa.Field(isin=_MSZoning_Options, nullable=True)
    LotFrontage: Series[int] = pa.Field(gt=0, nullable=True)
    LotArea: Series[int] = pa.Field(gt=0)
    Street: Series[str] = pa.Field(isin=_Street_Options)
    Alley: Series[str] = pa.Field(isin=_Alley_Options, nullable=True)
    LotShape: Series[str] = pa.Field(isin=_LotShape_Options)
    LandContour: Series[str] = pa.Field(isin=_LandCountour_Options)
    Utilities: Series[str] = pa.Field(isin=_Utilities_Options, nullable=True)
    LotConfig: Series[str] = pa.Field(isin=_LotConfig_Options)
    LandSlope: Series[str] = pa.Field(isin=_LandSlope_Options)
    Neighborhood: Series[str] = pa.Field(isin=_Neighborhood_Options)
    Condition1: Series[str] = pa.Field(isin=_Condition_Options)
    Condition2: Series[str] = pa.Field(isin=_Condition_Options)
    BldgType: Series[str] = pa.Field(isin=_BldgType_Options)
    HouseStyle: Series[str] = pa.Field(isin=_HouseStyle_Options)
    OverallQual: Series[int] = pa.Field(isin=range(1, 11))
    OverallCond: Series[int] = pa.Field(isin=range(1, 11))
    YearBuilt: Series[int] = pa.Field(gt=1799)
    YearRemodAdd: Series[int] = pa.Field(gt=1799)
    RoofStyle: Series[str] = pa.Field(isin=_RoofStyle_Options)
    RoofMatl: Series[str] = pa.Field(isin=_RoofMatl_Options)
    Exterior1st: Series[str] = pa.Field(isin=_Exterior_Options, nullable=True)
    Exterior2nd: Series[str] = pa.Field(isin=_Exterior_Options, nullable=True)
    MasVnrType: Series[str] = pa.Field(isin=_MasVnrTyp_Options, nullable=True)
    MasVnrArea: Series[int] = pa.Field(ge=0, nullable=True)
    ExterQual: Series[str] = pa.Field(isin=_Quality_Options)
    ExterCond: Series[str] = pa.Field(isin=_Quality_Options)
    Foundation: Series[str] = pa.Field(isin=_Foundation_Options)
    BsmtQual: Series[str] = pa.Field(isin=_QualityNA_Options, nullable=True)
    BsmtCond: Series[str] = pa.Field(isin=_QualityNA_Options, nullable=True)
    BsmtExposure: Series[str] = pa.Field(isin=_BsmtExposure_Options, nullable=True)
    BsmtFinType1: Series[str] = pa.Field(isin=_BsmtFinType_Options, nullable=True)
    BsmtFinSF1: Series[int] = pa.Field(ge=0, nullable=True)
    BsmtFinType2: Series[str] = pa.Field(isin=_BsmtFinType_Options, nullable=True)
    BsmtFinSF2: Series[int] = pa.Field(ge=0, nullable=True)
    BsmtUnfSF: Series[int] = pa.Field(ge=0, nullable=True)
    TotalBsmtSF: Series[int] = pa.Field(ge=0, nullable=True)
    Heating: Series[str] = pa.Field(isin=_Heating_Options)
    HeatingQC: Series[str] = pa.Field(isin=_Quality_Options)
    CentralAir: Series[str] = pa.Field(isin = _CentralAir_Options)
    Electrical: Series[str] = pa.Field(nullable=True)
    FirstFlrSF: Series[int] = pa.Field(gt=0)
    SecondFlrSF: Series[int] = pa.Field(ge=0)
    LowQualFinSF: Series[int] = pa.Field(ge=0)
    GrLivArea: Series[int] = pa.Field(gt=0)
    BsmtFullBath: Series[int] = pa.Field(ge=0, nullable=True)
    BsmtHalfBath: Series[int] = pa.Field(ge=0, nullable=True)
    FullBath: Series[int] = pa.Field(ge=0)
    HalfBath: Series[int] = pa.Field(ge=0)
    BedroomAbvGr: Series[int] = pa.Field(ge=0)
    KitchenAbvGr: Series[int] = pa.Field(ge=0)
    KitchenQual: Series[str] = pa.Field(isin=_Quality_Options, nullable=True)
    TotRmsAbvGrd: Series[int] = pa.Field(gt=0)
    Functional: Series[str] = pa.Field(isin=_Functional_Options, nullable=True)
    Fireplaces: Series[int] = pa.Field(ge=0)
    FireplaceQu: Series[str] = pa.Field(isin=_QualityNA_Options, nullable=True)
    GarageType: Series[str] = pa.Field(isin=_GarageType_Options, nullable=True)
    GarageYrBlt: Series[int] = pa.Field(gt=1799, nullable=True)
    GarageFinish: Series[str] = pa.Field(isin=_GarageFinish_Options, nullable=True)
    GarageCars: Series[int] = pa.Field(ge=0, nullable=True)
    GarageArea: Series[int] = pa.Field(ge=0, nullable=True)
    GarageQual: Series[str] = pa.Field(isin=_QualityNA_Options, nullable=True)
    GarageCond: Series[str] = pa.Field(isin=_QualityNA_Options, nullable=True)
    PavedDrive: Series[str] = pa.Field(isin=_PavedDrive_Options)
    WoodDeckSF: Series[int] = pa.Field(ge=0)
    OpenPorchSF: Series[int] = pa.Field(ge=0)
    EnclosedPorch: Series[int] = pa.Field(ge=0)
    ThreeSsnPorch: Series[int] = pa.Field(ge=0)
    ScreenPorch: Series[int] = pa.Field(ge=0)
    PoolArea: Series[int] = pa.Field(ge=0)
    PoolQC: Series[str] = pa.Field(isin=_QualityNA_Options, nullable=True)
    Fence: Series[str] = pa.Field(isin=_FenceQuality_Options, nullable=True)
    MiscFeature: Series[str] = pa.Field(isin=_MiscFeature_Options, nullable=True)
    MiscVal: Series[int] = pa.Field(ge=0)
    MoSold: Series[int] = pa.Field(isin=range(1,13))
    YrSold: Series[int] = pa.Field(gt=1799)
    SaleType: Series[str] = pa.Field(isin=_SaleType_Options, nullable=True)


class RawTargetSchema(pa.DataFrameModel):
    SalePrice: Optional[Series[int]] = pa.Field(gt=0)


@dataclass
class RawTestingData:
    df: pd.DataFrame

    def __post_init__(self):
        RawFeatureSchema.validate(self.df)


@dataclass
class RawTrainingData:
    df:pd.DataFrame

    def __post_init__(self):
        features_df = self.df.drop('SalePrice', axis=1)
        target_data = pd.DataFrame(self.df['SalePrice'])

        RawFeatureSchema.validate(features_df)
        RawTargetSchema.validate(target_data)


def load_training_data(filepath: str) -> RawTrainingData:
    """
    read csv into dataframe
    :param filepath: path to csv that should be loaded
    :return: df
    """

    df = pd.read_csv(filepath, dtype= _DTYPE_DICT|_TARGET_DTYPE).rename(columns={
        "1stFlrSF":"FirstFlrSF",
        "2ndFlrSF":"SecondFlrSF",
        "3SsnPorch":"ThreeSsnPorch"
    })

    return RawTrainingData(df)


def load_testing_data(filepath: str) -> RawTestingData:
    """
    read csv into dataframe
    :param filepath: path to csv that should be loaded
    :return: df
    """

    df = pd.read_csv(filepath, dtype= _DTYPE_DICT).rename(columns={
        "1stFlrSF": "FirstFlrSF",
        "2ndFlrSF": "SecondFlrSF",
        "3SsnPorch": "ThreeSsnPorch"
    })

    return RawTestingData(df)
