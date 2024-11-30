from dataclasses import dataclass

import lightgbm
import numpy as np
import pandas as pd
from sklearn import ensemble, linear_model  # type: ignore
import xgboost

import data_load
import data_prep
from hyperparameter_tuning import ModelHyperparameterMap


@dataclass
class PurePipeline:
    model_hyperparameter_map: ModelHyperparameterMap
    train_data_prep_inputs: data_prep.TrainDataPrepInputs
    testing_data: data_load.RawTestingData
    n_iter: int
    folds: int

    def run(self):
        # 1. perform data prep on training data
        train_data_prep_outputs = self.train_data_prep_inputs.train_data_prep()

        # 2. Measure performance of all of our different models
        champ_parameters = self.model_hyperparameter_map.measure_model_performance(train_data_prep_outputs, self.n_iter, self.folds)

        # 3. apply data prep from training to test data
        test_data_prep_inputs = data_prep.TestDataPrepInputs(self.testing_data, champ_parameters.categorical_encoders)
        test_X = test_data_prep_inputs.test_data_prep()

        # 4. predict test data
        # the model was trained against log transformed target. Invert log for predictions
        test_y = np.exp(champ_parameters.model.predict(test_X))
        test_y = pd.DataFrame(test_y, columns=[self.train_data_prep_inputs.target_variable], index=test_X.index)

        return test_y


def main():
    train_filepath = "predict_prices/data/train.csv"
    test_filepath = "predict_prices/data/test.csv"

    target_variable = 'SalePrice'

    # number of folds to cross-validate across
    folds = 4
    # model to cross validate
    lr = linear_model.LinearRegression()
    ohe_bool = False
    n_iter = 20

    # Initialize all model types
    rf = ensemble.RandomForestRegressor()
    rf_param_grid = {
        "n_estimators": tuple(range(10, 1000)),
        "min_samples_split": tuple(range(2, 10)),
        "min_samples_leaf": tuple(range(1, 10))
    }

    xgb = xgboost.XGBRegressor()
    xgb_param_grid = {
        "booster": ['gbtree', 'gblinear']
    }

    lgbm = lightgbm.LGBMRegressor(verbose=-1)
    lgbm_param_grid = {
        'max_depth': tuple(range(1, 100)),
        'num_leaves': tuple(range(1, 100000, 100))
    }

    en = linear_model.ElasticNet()
    lasso = linear_model.Lasso()
    ridge = linear_model.Ridge()
    br = linear_model.BayesianRidge()

    model_dict = {
        # TODO: Why wouldn't I convert these empty lists to empty dictionaries?
        lr: [],
        en: [],
        lasso: [],
        ridge: [],
        br: [],
        rf: rf_param_grid,
        xgb: xgb_param_grid,
        lgbm: lgbm_param_grid
    }
    hyper_param_map = ModelHyperparameterMap(model_dict)

    # 1. read in train and test data
    train_df = data_load.load_training_data(train_filepath)

    test_df = data_load.load_testing_data(test_filepath)
    train_data_prep_inputs = data_prep.TrainDataPrepInputs(train_df, ohe_bool, target_variable)
    pure_pipeline_inputs = PurePipeline(hyper_param_map,train_data_prep_inputs, test_df, n_iter, folds)

    # 2. run pure pipeline
    test_y = pure_pipeline_inputs.run()

    # 3. output predictions to csv
    test_y.to_csv('prediction.csv')


if __name__ == '__main__':
    main()
