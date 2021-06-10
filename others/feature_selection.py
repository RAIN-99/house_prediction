from category_encoders import CountEncoder
import dill
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
desired_width=320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns',50)
full_data=pd.read_csv("processed_data.csv")
features,target=full_data.drop(columns='Цена'),full_data['Цена']


class CollinearFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, min_corr_with_target, max_allowed_corr_between_features):
        self.min_corr_with_target = min_corr_with_target
        self.max_allowed_corr_between_features = max_allowed_corr_between_features
        self.selected_features = None

    def fit(self, X, Y):
        categorical_features = X.select_dtypes(object).columns.tolist()

        corr_with_target_data = X.corrwith(Y).abs()
        corr_with_target_data = corr_with_target_data[corr_with_target_data >= self.min_corr_with_target]

        if self.max_allowed_corr_between_features == 1:
            selected_features = corr_with_target_data.index.tolist() + categorical_features
        else:
            sorted_by_corr_features = corr_with_target_data.sort_values().index
            correlation_matrix = X[sorted_by_corr_features].corr()
            for feature in sorted_by_corr_features:
                if correlation_matrix.loc[:, feature].drop(
                        index=feature).abs().max() > self.max_allowed_corr_between_features:
                    correlation_matrix = correlation_matrix.drop(columns=feature).drop(index=feature)
            selected_features = correlation_matrix.columns.tolist() + categorical_features

        self.selected_features = selected_features
        return self

    def transform(self, X):
        X_transformed = X[self.selected_features]
        return X_transformed


def get_baseline_for_feature_selector(feature_selector, features=features, target=target):
    features_selected = feature_selector.fit_transform(features, target)
    features_selected = fill_missing_values(features_selected)
    cv_results = get_cv_results(features_selected, target)
    return cv_results, features_selected.shape[1]


def fill_missing_values(data):
    categorical_columns = data.select_dtypes(object).columns.tolist()
    data[categorical_columns] = data[categorical_columns].fillna("MISSING")

    numerical_columns = data.select_dtypes('number').columns.tolist()
    data[numerical_columns] = data[numerical_columns].fillna(-999)
    return data


def get_cv_results(features, target):
    categorical_cols = features.select_dtypes(object).columns
    pipeline = Pipeline(steps=[('encoder', CountEncoder(cols=categorical_cols,
                                                        min_group_size=1,
                                                        handle_unknown=0)),
                               ('regressor', RandomForestRegressor(random_state=5, max_features='sqrt'))])

    kfold_generator = KFold(n_splits=10, shuffle=True, random_state=5)
    scoring = ['neg_mean_absolute_error', 'r2', 'neg_root_mean_squared_error']
    cv_results = cross_validate(pipeline, features, target, cv=kfold_generator.split(features, target),
                                n_jobs=5, scoring=scoring)
    return cv_results


optimization_results = []
for max_allowed_corr_between_features in tqdm(np.arange(1, 0, -0.1)):
    feature_selector = CollinearFeatureSelector(min_corr_with_target=0,
                                                max_allowed_corr_between_features=max_allowed_corr_between_features)
    cv_results, features_num = get_baseline_for_feature_selector(feature_selector)
    optimization_results.append({
        'min_corr_with_target' : 0,
        'max_allowed_corr_between_features' : max_allowed_corr_between_features,
        'features_num' : features_num,
        'mean_fit_time' : cv_results['fit_time'].mean(),
        'MAE' : cv_results['test_neg_mean_absolute_error'].mean(),
        'RMSE' : cv_results['test_neg_root_mean_squared_error'].mean(),
        'R-squared' : cv_results['test_r2'].mean()
    })

best_result_max=pd.DataFrame(optimization_results, columns=optimization_results[0].keys())
print(best_result_max)

feature_selector = CollinearFeatureSelector(min_corr_with_target=0, max_allowed_corr_between_features=0.3)
features_selected = feature_selector.fit_transform(features, target)
dill.dump(feature_selector, open('./feature_selector.pkl', 'wb'))