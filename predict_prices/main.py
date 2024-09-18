import lightgbm
import numpy as np
import pandas as pd
from sklearn import ensemble, linear_model # type: ignore
import xgboost

import cross_validation
import data_prep

def main():
    train_fp = "predict_prices/data/train.csv"
    test_fp = "predict_prices/data/test.csv"
    # target variable
    tgt = 'SalePrice'
    # temporary variable to remove categoricals
    include_categoricals = False
    # number of folds to cross-validate across
    folds = 4
    # model to cross validate
    lr = linear_model.LinearRegression()
    ohe = False
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
    train_df = data_prep.load_data(train_fp)
    train_df = data_prep.eda_clean(train_df)
    train_df, _ = data_prep.clean_after_eda(train_df)
    ord_enc = data_prep.ordinal_encode(train_df)
    train_df = data_prep.ordinal_transform(train_df, ord_enc)
    cat_enc = data_prep.categorical_encoder(train_df, ohe)
    train_df = data_prep.categorical_transform(train_df, cat_enc)
    train_X, train_y = data_prep.split_x_y(train_df, tgt)
    train_X = train_X if include_categoricals else data_prep.drop_categoricals(train_X)

    # 2. Measure performance of all of our different models
    lr.fit(train_X, train_y)
    lr_param_scores = cross_validation.save_model_performance_parameters(lr, train_X, train_y, folds, ord_enc, cat_enc)

    en.fit(train_X, train_y)
    en_param_scores = cross_validation.save_model_performance_parameters(en, train_X, train_y, folds, ord_enc, cat_enc)

    lasso.fit(train_X, train_y)
    lasso_param_scores = cross_validation.save_model_performance_parameters(lasso, train_X, train_y, folds, ord_enc,
                                                                            cat_enc)
    ridge.fit(train_X, train_y)
    ridge_param_scores = cross_validation.save_model_performance_parameters(ridge, train_X, train_y, folds, ord_enc,
                                                                            cat_enc)
    br.fit(train_X, train_y)
    br_param_scores = cross_validation.save_model_performance_parameters(br, train_X, train_y, folds, ord_enc, cat_enc)

    rf = cross_validation.bayes_cross_validation(rf, train_X, train_y, rf_param_grid, n_iter)
    rf_param_scores = cross_validation.save_model_performance_parameters(rf, train_X, train_y, folds, ord_enc, cat_enc)

    xgb = cross_validation.bayes_cross_validation(xgb, train_X, train_y, xgb_param_grid, n_iter)
    xgb_param_scores = cross_validation.save_model_performance_parameters(xgb, train_X, train_y, folds, ord_enc, cat_enc)

    lgbm = cross_validation.bayes_cross_validation(lgbm, train_X, train_y, lgbm_param_grid, n_iter)
    lgbm_param_scores = cross_validation.save_model_performance_parameters(lgbm, train_X, train_y, folds, ord_enc, cat_enc)

    # 3. Compare performance and select best model
    models = [lr_param_scores, rf_param_scores, xgb_param_scores, lgbm_param_scores, en_param_scores, lasso_param_scores]
    param_scores = pd.DataFrame(models).sort_values('score', ascending=False)

    print(param_scores[['model', 'score', 'standard_dev']])

    # take only first row in case of ties (I don't care which model if they're tied)
    champ = param_scores[param_scores.score == param_scores.score.max()].reset_index()
    champ_model = champ.model[0]
    champ_ord_enc = champ.ordinal_encoder[0]
    champ_cat_enc = champ.categorical_encoder[0]

    # 4. Read in test data and apply data prep from step 1
    test_df = data_prep.load_data(test_fp)
    test_df = data_prep.eda_clean(test_df)
    test_df, _ = data_prep.clean_after_eda(test_df)
    test_df = data_prep.ordinal_transform(test_df, champ_ord_enc)
    test_df = data_prep.categorical_transform(test_df, champ_cat_enc)
    test_X = test_df if include_categoricals else data_prep.drop_categoricals(test_df)

    # 5. predict test data and output to csv
    # the model was trained against log transformed target. Invert log for predictions
    test_y = np.exp(champ_model.predict(test_X))
    print(test_X.head())

    test_y = pd.DataFrame(test_y, columns=['SalePrice'], index=test_X.index)

    test_y.to_csv('prediction.csv')


if __name__ == '__main__':
    main()
