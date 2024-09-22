import lightgbm
import numpy as np
import pandas as pd
from sklearn import ensemble, linear_model # type: ignore
import xgboost

import cross_validation
import data_prep

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

    # 1. read in training data, perform data prep
    train_df = data_prep.load_data(train_filepath)
    train_data_prep_inputs = data_prep.TrainDataPrepInputs(train_df, ohe_bool, target_variable, include_categoricals)
    train_data_prep_outputs = data_prep.train_data_prep(train_data_prep_inputs)


    # 2. Measure performance of all of our different models
    lr.fit(train_data_prep_outputs.train_data.X, train_data_prep_outputs.train_data.y)
    lr_param_scores = cross_validation.save_model_performance_parameters(lr, folds, train_data_prep_outputs)

    en.fit(train_data_prep_outputs.train_data.X, train_data_prep_outputs.train_data.y)
    en_param_scores = cross_validation.save_model_performance_parameters(en, folds, train_data_prep_outputs)

    lasso.fit(train_data_prep_outputs.train_data.X, train_data_prep_outputs.train_data.y)
    lasso_param_scores = cross_validation.save_model_performance_parameters(lasso, folds, train_data_prep_outputs)

    ridge.fit(train_data_prep_outputs.train_data.X, train_data_prep_outputs.train_data.y)
    ridge_param_scores = cross_validation.save_model_performance_parameters(ridge, folds, train_data_prep_outputs)

    br.fit(train_data_prep_outputs.train_data.X, train_data_prep_outputs.train_data.y)
    br_param_scores = cross_validation.save_model_performance_parameters(br, folds, train_data_prep_outputs)

    rf = cross_validation.bayes_cross_validation(rf, train_data_prep_outputs.train_data, rf_param_grid, n_iter)
    rf_param_scores = cross_validation.save_model_performance_parameters(rf, folds, train_data_prep_outputs)

    xgb = cross_validation.bayes_cross_validation(xgb, train_data_prep_outputs.train_data, xgb_param_grid, n_iter)
    xgb_param_scores = cross_validation.save_model_performance_parameters(xgb, folds, train_data_prep_outputs)

    lgbm = cross_validation.bayes_cross_validation(lgbm, train_data_prep_outputs.train_data, lgbm_param_grid, n_iter)
    lgbm_param_scores = cross_validation.save_model_performance_parameters(lgbm, folds, train_data_prep_outputs)

    # 3. Compare performance and select best model
    models = [lr_param_scores, rf_param_scores, xgb_param_scores, lgbm_param_scores, en_param_scores, lasso_param_scores,
              ridge_param_scores, br_param_scores]
    param_scores = pd.DataFrame(models).sort_values('score', ascending=False)

    print(param_scores[['model', 'score', 'standard_dev']])

    # take only first row in case of ties (I don't care which model if they're tied)
    champ = param_scores[param_scores.score == param_scores.score.max()].reset_index()
    champ_model = champ.model[0]
    champ_ord_enc = champ.ordinal_encoder[0]
    champ_cat_enc = champ.categorical_encoder[0]

    # 4. Read in test data and apply data prep from step 1
    test_df = data_prep.load_data(test_filepath)
    champ_categorical_encoders = data_prep.CategoricalEncoders(champ_cat_enc, champ_ord_enc)
    test_data_prep_inputs = data_prep.TestDataPrepInputs(test_df, champ_categorical_encoders, include_categoricals)
    test_X = data_prep.test_data_prep(test_data_prep_inputs)

    # 5. predict test data and output to csv
    # the model was trained against log transformed target. Invert log for predictions
    test_y = np.exp(champ_model.predict(test_X))
    print(test_X.head())

    test_y = pd.DataFrame(test_y, columns=['SalePrice'], index=test_X.index)

    test_y.to_csv('prediction.csv')


if __name__ == '__main__':
    main()
