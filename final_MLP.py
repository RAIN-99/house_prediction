from category_encoders import CountEncoder
import dill
import numpy as np
import pandas as pd
import shap
from sklearn import preprocessing
from sklearn.model_selection import KFold, GridSearchCV, cross_validate,train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import keras
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping
from keras import backend
from keras.wrappers.scikit_learn import KerasRegressor
full_data=pd.read_csv("../data/processed_data.csv")
full_data=full_data[['Общая площадь',"Цена","Долгота","Состояние","Широта","Количество комнат","Всего этажей","Серия","Этаж","Год постройки","Санузел","Мебель","Отопление","Тип предложения"]]
X_train,X_test,y_train,y_test=pd.read_csv('X_train.csv'),pd.read_csv('X_test.csv'),pd.read_csv('y_train.csv'),pd.read_csv('y_test.csv')
print(X_train.shape,X_test.shape)

model = keras.models.Sequential()
model.add(keras.layers.Dense(512, activation='relu', input_shape=(len(X_train.columns),),activity_regularizer=l1_l2(l2=0.001)))
model.add(keras.layers.Dropout(0))
model.add(keras.layers.Dense(256, activation='relu', activity_regularizer=l1_l2(l2=0.001)))
model.add(keras.layers.Dropout(0))
model.add(keras.layers.Dense(128, activation='relu', activity_regularizer=l1_l2(l2=0.001)))
model.add(keras.layers.Dropout(0))
model.add(keras.layers.Dense(64, activation='relu', activity_regularizer=l1_l2(l2=0.001)))
model.add(keras.layers.Dropout(0))
model.add(keras.layers.Dense(32, activation='relu', activity_regularizer=l1_l2(l2=0.001)))
model.add(keras.layers.Dropout(0))
model.add(keras.layers.Dense(16, activation='relu', activity_regularizer=l1_l2(l2=0.001)))
model.add(keras.layers.Dropout(0))
model.add(keras.layers.Dense(8, activation='relu', activity_regularizer=l1_l2(l2=0.001)))
model.add(keras.layers.Dense(1, activation='linear'))
model.compile(optimizer='Adam', loss="mean_squared_error",metrics=[keras.metrics.RootMeanSquaredError(), 'mean_squared_logarithmic_error'])
model.save_weights('model.h5')

# categorical_columns = X_train.select_dtypes(object).columns
# CE=CountEncoder(cols=categorical_columns,handle_unknown=0,min_group_size=1)
# X_train=CE.fit_transform(X_train)
scaler = preprocessing.MinMaxScaler()
X_train=scaler.fit_transform(X_train)
# X_test=CE.transform(X_test)
X_test=scaler.transform(X_test)
print(pd.DataFrame(X_train).head())
def plot_metric(history,metric,n):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.figure()
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    #plt.title('Training and validation '+ metric + n)
    plt.xlabel("Эпоха",fontsize=22)
    plt.ylabel("Корень из среднеквадратической ошибки",fontsize=22)
    plt.xticks(size=22)
    plt.yticks(size=22)
    plt.legend(["ошибка на тренировке","ошибка на валидации"],prop={'size':22})
    plt.show()
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,shuffle=True,test_size=0.1)
print(X_train.shape,X_val.shape)
inputs=np.concatenate((X_train,X_val),axis=0)
targets=np.concatenate((y_train,y_val),axis=0)
e_s = EarlyStopping(monitor='val_root_mean_squared_error', patience=250)
kfold_generator = KFold(n_splits=10, shuffle=True, random_state=5)
fold_no=1
RMSE_per_fold=[]
MSLE_per_fold=[]
number_of_epochs=[]
for train,test in kfold_generator.split(inputs,targets):
    model.load_weights('model.h5')
    history=model.fit(inputs[train],targets[train],verbose=0,batch_size=128,validation_data=(inputs[test],targets[test]),epochs=5000,shuffle=True,callbacks=[e_s])
    scores=model.evaluate(inputs[test],targets[test],verbose=0)
    RMSE_per_fold.append(scores[1])
    MSLE_per_fold.append(scores[2])
    number_of_epochs.append(len(history.history['root_mean_squared_error']))
    print(f'Score for fold {fold_no}: {model.metrics_names[1]} of {scores[1]}; {model.metrics_names[2]} of {scores[2]}; epochs:{number_of_epochs}')
    fold_no+=1
    plot_metric(history, "root_mean_squared_error",str(fold_no))
    plot_metric(history, "mean_squared_logarithmic_error",str(fold_no))
pd.DataFrame(list(zip(RMSE_per_fold,MSLE_per_fold,number_of_epochs)),columns=['RMSE','MSLE','epochs']).to_excel("../data/cv_results_MLP.xlsx")
X_test['prediction']=(model.predict(X_test))
X_test.to_csv('y_pred_XGboost.csv')
print(pd.DataFrame(RMSE_per_fold).mean)
plt.show()

