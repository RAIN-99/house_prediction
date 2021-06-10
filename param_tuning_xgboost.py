from category_encoders import CountEncoder
import dill
import numpy as np
from sklearn import preprocessing
import pandas as pd
import shap
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBRegressor
full_data=pd.read_csv("../data/processed_data.csv")
print(full_data.columns)
features, target = full_data.drop(columns=['Цена']), full_data['Цена']
print(features.shape)
print(target.shape)
def fill_missing_values(data):
    categorical_columns = data.select_dtypes(object).columns.tolist()
    data[categorical_columns] = data[categorical_columns].fillna("MISSING")

    numerical_columns = data.select_dtypes('number').columns.tolist()
    data[numerical_columns] = data[numerical_columns].fillna(-999)
    return data
features = fill_missing_values(features)
print(features.shape)
print(target.shape)
features_important = features#[['Общая площадь', 'Долгота', 'Широта', 'Состояние', 'Количество комнат']]
categorical_columns = features_important.select_dtypes(object).columns
scoring=['neg_mean_absolute_error', 'r2', 'neg_root_mean_squared_error',"neg_mean_squared_log_error"]
pipeline = Pipeline(steps=[('encoder', CountEncoder(cols=categorical_columns,
                                                    min_group_size=1,
                                                    handle_unknown=0)),
                           ('regressor', XGBRegressor(n_estimators=1000,
                                                      verbosity=0,
                                                      reg_lambda=0.4,
                                                      reg_alpha=0.4,
                                                      sample_type='uniform',
                                                      rate_drop=0.1,
                                                      booster='gbtree',
                                                      subsample=1,
                                                      colsample_bylevel=1,
                                                      colsample_bynode=1,
                                                      colsample_bytree=1,
                                                      gamma=0.1,
                                                      importance_type='gain',
                                                      max_delta_step=0,
                                                      min_child_weight=1,
                                                      missing=None,
                                                      n_jobs=1,
                                                      objective='reg:squarederror',
                                                      max_depth=4,
                                                      learning_rate=0.07,
                                                      random_state=5))])
params = {
    'regressor__n_estimators' : [1000],
    'regressor__max_depth' : [4],
    'regressor__learning_rate' : [0.07],
    'regressor__reg_lambda':[0.4],
    'regressor__reg_alpha':[0.4],
    'regressor__gamma':[0.1],
    'regressor__rate_drop':[0.1],
    'regressor__subsample':[1]
}
kfold_generator = KFold(n_splits=10, shuffle=True, random_state=5)
search = GridSearchCV(pipeline, params, cv=kfold_generator, scoring=scoring, n_jobs=4, verbose=100, refit='neg_root_mean_squared_error')
search.fit(features_important,target)
search_res=pd.DataFrame(search.cv_results_)
print(search_res.sort_values('mean_test_neg_root_mean_squared_error'))
best_model = search.best_estimator_
dill.dump(best_model, open('_model.pkl', 'wb'))
print(best_model)