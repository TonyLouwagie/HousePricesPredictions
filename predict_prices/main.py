from dataclasses import dataclass
from typing import Iterable

import lightgbm
import numpy as np
import pandas as pd
from sklearn import ensemble, linear_model  # type: ignore
import xgboost

import cross_validation
import data_prep


@dataclass
class ChampionModelParameters:
    model: any
    categorical_encoders: data_prep.CategoricalEncoders

@dataclass
class ModelParameterMap:
    model_params_map: dict[any, dict[str, Iterable]]

    def measure_model_performance(
            self,
            train_data_prep_outputs: data_prep.TrainDataPrepOutputs,
            n_iter: int,
            folds: int
    ) -> ChampionModelParameters:

        models = []

        for i in self.model_params_map:
            param_grid = self.model_params_map[i]

            if len(param_grid) > 0:
                best_model = cross_validation.bayes_cross_validation(i, train_data_prep_outputs.train_data, param_grid,
                                                                     n_iter)
                best_model_param_scores = cross_validation.save_model_performance_parameters(best_model, folds,
                                                                                             train_data_prep_outputs)
            else:
                i.fit(train_data_prep_outputs.train_data.X, train_data_prep_outputs.train_data.y)
                best_model_param_scores = cross_validation.save_model_performance_parameters(i, folds,
                                                                                             train_data_prep_outputs)

            models.append(best_model_param_scores)

            param_scores = pd.DataFrame(models).sort_values('score', ascending=False)

            print(param_scores[['model', 'score', 'standard_dev']])

            # take only first row in case of ties (I don't care which model if they're tied)
            champ = param_scores[param_scores.score == param_scores.score.max()].reset_index()

            champ_model = champ.model[0]
            champ_ord_enc = champ.ordinal_encoder[0]
            champ_cat_enc = champ.categorical_encoder[0]

            champ_parameters = ChampionModelParameters(champ_model, data_prep.CategoricalEncoders(champ_ord_enc, champ_cat_enc))

        return champ_parameters


def main():
    train_filepath = "predict_prices/data/train.csv"
    test_filepath = "predict_prices/data/test.csv"

    target_variable = 'SalePrice'
    # temporary variable to remove categoricals
    include_categoricals = False
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
        lr: [],
        en: [],
        lasso: [],
        ridge: [],
        br: [],
        rf: rf_param_grid,
        xgb: xgb_param_grid,
        lgbm: lgbm_param_grid
    }
    model_param_map = ModelParameterMap(model_dict)

    # 1. read in training data, perform data prep
    train_df = data_prep.load_data(train_filepath)
    train_data_prep_inputs = data_prep.TrainDataPrepInputs(train_df, ohe_bool, target_variable, include_categoricals)
    train_data_prep_outputs = train_data_prep_inputs.train_data_prep()

    # 2. Measure performance of all of our different models
    champ_parameters = model_param_map.measure_model_performance(train_data_prep_outputs, n_iter, folds)

    # 3. Read in test data and apply data prep from step 1
    test_df = data_prep.load_data(test_filepath)
    test_data_prep_inputs = data_prep.TestDataPrepInputs(test_df, champ_parameters.categorical_encoders, include_categoricals)
    test_X = test_data_prep_inputs.test_data_prep()

    # 4. predict test data and output to csv
    # the model was trained against log transformed target. Invert log for predictions
    test_y = np.exp(champ_parameters.model.predict(test_X))
    test_y = pd.DataFrame(test_y, columns=[target_variable], index=test_X.index)
    test_y.to_csv('prediction.csv')


if __name__ == '__main__':
    main()
