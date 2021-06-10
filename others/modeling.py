from category_encoders import CountEncoder
import dill
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost.sklearn import XGBRegressor
from tqdm import tqdm

full_data=pd.read_csv("processed_data.csv")

def fill_missing_values(data):
    categorical_columns = data.select_dtypes(object).columns.tolist()
    data[categorical_columns] = data[categorical_columns].fillna("MISSING")

    numerical_columns = data.select_dtypes('number').columns.tolist()
    data[numerical_columns] = data[numerical_columns].fillna(-999)
    return data

full_data = fill_missing_values(full_data)
features,target=full_data.drop(columns='Цена'),full_data['Цена']

feature_selector = dill.load(open('./feature_selector.pkl', 'rb'))
features = feature_selector.transform(features)
print(features.columns)
categorical_columns = full_data.select_dtypes(object).columns
# # pipeline = Pipeline(steps=[('encoder', CountEncoder(cols=categorical_columns,
# #                                                     min_group_size=1,
# #                                                     handle_unknown=0)),
# #                            ('regressor', XGBRegressor(random_state=5))])
# #
# # params = {
# #     'regressor__n_estimators' : [100, 200, 500, 1000],
# #     'regressor__max_depth' : [1, 4, 8, 10],
# #     'regressor__learning_rate' : [0.001, 0.01, 0.1, 1]
# # }
# # kfold_generator = KFold(n_splits=10, shuffle=True, random_state=5)
# # scoring=['neg_mean_absolute_error', 'r2', 'neg_root_mean_squared_error']
# # search = GridSearchCV(pipeline, params, cv=kfold_generator, scoring=scoring, n_jobs=6, verbose=100, refit='r2')
# # search.fit(features, target)
# # search_results = pd.DataFrame(search.cv_results_)
# # search_results.to_csv('./optimization_results.csv', index=False)
# # pd.options.display.max_colwidth = None
# # print(search_results.sort_values('mean_test_r2', ascending=False).loc[:, search_results.columns.str.contains('params|^mean')][0:10])
#
# best_model = search.best_estimator_
# dill.dump(best_model, open('./final_model.pkl', 'wb'))

best_model = dill.load(open('./final_model.pkl', 'rb'))

def plot_feature_importances(pipeline, features, target):
    from matplotlib import pyplot as plt
    import pandas as pd
    import seaborn as sns

    pipeline.fit(features, target)

    feature_importances_df = pd.DataFrame()
    feature_importances_df['feature'] = features.columns
    feature_importances_df['importance'] = pipeline['regressor'].feature_importances_
    feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importances_df[0:30])
    plt.show()
plot_feature_importances(best_model, features, target)
explainer = shap.TreeExplainer(best_model['regressor'])
features_encoded = best_model['encoder'].transform(features)
shap_values = explainer.shap_values(features_encoded)
shap.summary_plot(shap_values, features_encoded)
plt.show()
shap.summary_plot(shap_values, features_encoded, plot_type='bar')
plt.show()
