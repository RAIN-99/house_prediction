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
from sklearn import tree
import graphviz
full_data=pd.read_csv("../../data/processed_data.csv")
X_train,X_test,y_train,y_test=pd.read_csv('../X_train.csv'),pd.read_csv('../X_test.csv'),pd.read_csv('../y_train.csv'),pd.read_csv('../y_test.csv')
feature_names = [i for i in full_data.columns if full_data[i].dtype in [np.int64, np.int64]]
X = full_data[feature_names]
print(y_test.mean())
model = dill.load(open('../final_model_XGboost.pkl','rb'))

###
# ИНТЕРПРЕТАЦИЯ
###
row_to_show=5
data_for_prediction=y_test.iloc[row_to_show]
data_for_prediction_array=data_for_prediction.values.reshape(-1,1)
def explanation_of_single_example():
    explainer = shap.Explainer(model['regressor'])
    shap_values=explainer(X_test)
    shap.plots.waterfall(shap_values[2],max_display=20)#6
    pass
explanation_of_single_example()
def explanation_of_all_features():
    explainer = shap.TreeExplainer(model['regressor'])
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values,X_test,max_display=20)#6
    shap.summary_plot(shap_values, X_test, plot_type='bar')
    pass

def force_plots():
    explainer = shap.Explainer(model['regressor'])
    shap_values = explainer(X_test)
    shap.initjs()
    shap.force_plot(shap_values[1],data_for_prediction,matplotlib=True)
    plt.show()
    pass

def dependence_contribution_plots(x,y,X_test):
    explainer = shap.TreeExplainer(model['regressor'])
    shap_values = explainer.shap_values(X_test)
    X_test=data_decoder(X_test,x,y)
    shap.dependence_plot(x,shap_values,X_test,interaction_index=y,cmap=plt.get_cmap('jet'))
    pass

def data_decoder(data,x,y):
    decoded_data=data.copy()
    if y or x =='Тип предложения':
        if y =='Тип предложения':
            decoded_data[y].replace({3143:'От собственника',2623:'От агента'},inplace=True)
        if x =='Тип предложения':
            decoded_data[x].replace({3143:'От собственника',2623:'От агента'},inplace=True)
    if y or x == 'Состояние':
        if y =='Состояние':
            decoded_data[y].replace({939:'ПСО',1510:'Хорошее',1372:"Евроремонт",338:"Среднее",194:"Требует ремонта",103:"Черновая отделка",53:"Недостроенное",1257:"Пропуск"},inplace=True)
        if x =='Состояние':
            decoded_data[x].replace({939:'ПСО',1510:'Хорошее',1372:"Евроремонт",338:"Среднее",194:"Требует ремонта",103:"Черновая отделка",53:"Недостроенное",1257:"Пропуск"},inplace=True)
    if y or x == 'Отопление':
        if y == 'Отопление':
            decoded_data[y].replace({4104:"Центральное",272:"На газе",277:"Электрическое",220:"Автономное",26:"Сменшанное",867:"Пропуск"},inplace=True)
        if x == 'Отопление':
            decoded_data[x].replace({4104:"Центральное",272:"На газе",277:"Электрическое",220:"Автономное",26:"Сменшанное",867:"Пропуск"},inplace=True)
    if y or x == 'Санузел':
        if y == 'Санузел':
            decoded_data[y].replace({1775:"Раздельный",1229:"Совмещенный",409:"2 с/у и более",2353:"Пропуск"},inplace=True)
        if x == 'Санузел':
            decoded_data[x].replace({1775:"Раздельный",1229:"Совмещенный",409:"2 с/у и более",2353:"Пропуск"},inplace=True)
    if y or x =='Мебель':
        if y == 'Мебель':
            decoded_data[y].replace({1148:"Частично меблирована",771:"Полностью меблирована",716:"Пустая",3131:"Пропуск"},inplace=True)
        if x == 'Мебель':
            decoded_data[x].replace({1148:"Частично меблирована",771:"Полностью меблирована",716:"Пустая",3131:"Пропуск"},inplace=True)
    if x or y == 'Серия':
        if y == 'Серия':
            decoded_data[y].replace({1189:"Инд.планировка",2731:"Элитка",530:"105-серия",425:"104-серия",519:"106-серия",165:"Хрущевка",74:"Малосемейка",56:"Сталинка",41:"108-серия",36:"Пентхаус"},inplace=True)
        if x == 'Серия':
            decoded_data[x].replace({1189:"Инд.планировка",2731:"Элитка",530:"105-серия",425:"104-серия",519:"106-серия",165:"Хрущевка",74:"Малосемейка",56:"Сталинка",41:"108-серия",36:"Пентхаус"},inplace=True)
    return decoded_data


dependence_contribution_plots('Общая площадь','Тип предложения',X_test)
dependence_contribution_plots('Общая площадь','Состояние',X_test)



def dependence_contribution_plots(x,y,X_test):
    explainer = shap.TreeExplainer(model['regressor'])
    shap_values = explainer.shap_values(X_test)
    shap.dependence_plot(x,shap_values,X_test,interaction_index=y,cmap=plt.get_cmap('jet'))
    pass

dependence_contribution_plots('Общая площадь','Количество комнат',X_test)


plt.show()