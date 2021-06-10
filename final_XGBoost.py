from category_encoders import CountEncoder
import dill
import numpy as np
from sklearn import preprocessing
import pandas as pd
import shap
from sklearn.model_selection import KFold, GridSearchCV, cross_validate,train_test_split
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBRegressor
from xgboost import plot_tree
from matplotlib import  pyplot
import os
import matplotlib.pyplot as plt
full_data=pd.read_csv("../data/processed_data.csv")
# full_data=full_data[['Общая площадь',"Цена","Долгота","Состояние","Широта","Количество комнат","Всего этажей","Серия","Этаж","Год постройки","Санузел","Мебель","Отопление","Тип предложения"]]
# features, target = full_data.drop(columns=['Цена']), full_data['Цена']
# X_train,X_test,y_train,y_test=train_test_split(features,target,shuffle=True,test_size=0.02)
# pd.DataFrame(X_train).to_csv('X_train.csv',index=False)
# pd.DataFrame(X_test).to_csv('X_test.csv',index=False)
# pd.DataFrame(y_train).to_csv('y_train.csv',index=False)
# pd.DataFrame(y_test).to_csv('y_test.csv',index=False)
X_train,X_test,y_train,y_test=pd.read_csv('X_train.csv'),pd.read_csv('X_test.csv'),pd.read_csv('y_train.csv'),pd.read_csv('y_test.csv')
print(X_train.shape,X_test.shape)
categorical_columns = X_train.select_dtypes(object).columns
print(categorical_columns)
scoring=['r2', 'neg_root_mean_squared_error',"neg_mean_squared_log_error"]
pipeline = Pipeline(steps=[('encoder', CountEncoder(cols=categorical_columns,
                                                    min_group_size=1,
                                                    handle_unknown=0)),
                           ('regressor', XGBRegressor(n_estimators=1000,
                                                      verbosity=0,
                                                      reg_lambda=0.4,
                                                      reg_alpha=0.4,
                                                      sample_type='uniform',
                                                      rate_drop=0.1,
                                                      base_score=0.5,
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
                                                      n_jobs=4,
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
search = GridSearchCV(pipeline, params, cv=kfold_generator, scoring=scoring, n_jobs=4, verbose=0, refit='neg_root_mean_squared_error')
search.fit(X_train,y_train)
search_res=pd.DataFrame(search.cv_results_)
search_res.to_excel('../data/optimization.xlsx')
print(search_res[['mean_test_neg_root_mean_squared_error',"mean_test_neg_mean_squared_log_error"]])
best_model = search.best_estimator_
X_test['prediction']=(best_model.predict(X_test))
X_test.to_csv('y_pred_XGboost.csv',index=False)
dill.dump(best_model, open('final_model_XGboost.pkl', 'wb'))
os.environ['PATH']+=os.pathsep + 'C:/Program Files/Graphviz/bin/'
plot_tree(best_model['regressor'])
plt.savefig('destination_path.eps', format='eps',dpi=1200)
#plt.show()