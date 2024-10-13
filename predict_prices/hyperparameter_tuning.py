from dataclasses import dataclass
from typing import Iterable

import pandas as pd

import data_prep
import cross_validation


@dataclass
class ModelParameters:
    model: any
    categorical_encoders: data_prep.CategoricalEncoders


@dataclass
class ModelHyperparameterMap:
    hyper_params_map: dict[any, dict[str, Iterable]]

    def measure_model_performance(
            self,
            train_data_prep_outputs: data_prep.TrainDataPrepOutputs,
            n_iter: int,
            folds: int
    ) -> ModelParameters:

        models = []

        for i in self.hyper_params_map:
            param_grid = self.hyper_params_map[i]

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

        champ_parameters = ModelParameters(champ_model, data_prep.CategoricalEncoders(champ_ord_enc, champ_cat_enc))

        return champ_parameters
