import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
X_train,X_test,y_train,y_test=pd.read_csv('../X_train.csv'),pd.read_csv('../X_test.csv'),pd.read_csv('../y_train.csv'),pd.read_csv('../y_test.csv')
# X_train['Состояние'].replace({15:38},inplace=True)
# X_test['Состояние'].replace({15:38},inplace=True)
# X_train['Состояние'].replace({38:53},inplace=True)
# X_test['Состояние'].replace({38:53},inplace=True)
# print(X_train['Отопление'].count())
# print(X_test['Отопление'].count())
# X_train['Отопление'].replace({5:861},inplace=True)
# X_test['Отопление'].replace({5:861},inplace=True)
# X_train['Отопление'].replace({1:861},inplace=True)
# X_test['Отопление'].replace({1:861},inplace=True)
# X_train['Отопление'].replace({861:867},inplace=True)
# X_test['Отопление'].replace({861:867},inplace=True)
# print(X_train[X_train["Санузел"]==12].value_counts())
# X_train['Санузел'].replace({12:2341},inplace=True)
# X_test['Санузел'].replace({12:2341},inplace=True)
# X_train['Санузел'].replace({2341:2353},inplace=True)
# X_test['Санузел'].replace({2341:2353},inplace=True)
#
# X_train['Серия'].replace({208:311},inplace=True)
# X_test['Серия'].replace({208:311},inplace=True)
# X_train['Серия'].replace({311:519},inplace=True)
# X_test['Серия'].replace({311:519},inplace=True)
#
# X_train['Серия'].replace({57:473},inplace=True)
# X_test['Серия'].replace({57:473},inplace=True)
# X_train['Серия'].replace({473:530},inplace=True)
# X_test['Серия'].replace({473:530},inplace=True)
#
# X_train['Серия'].replace({57:473},inplace=True)
# X_test['Серия'].replace({57:473},inplace=True)
# X_train['Серия'].replace({473:530},inplace=True)
# X_test['Серия'].replace({473:530},inplace=True)
#
# X_train['Серия'].replace({51:374},inplace=True)
# X_test['Серия'].replace({51:374},inplace=True)
# X_train['Серия'].replace({374:425},inplace=True)
# X_test['Серия'].replace({374:425},inplace=True)

# X_train['Серия'].replace({7:2728},inplace=True)
# X_test['Серия'].replace({7:2728},inplace=True)
# X_train['Серия'].replace({2735:2731},inplace=True)
# X_test['Серия'].replace({2735:2731},inplace=True)
# #
# print(X_train[X_train['Серия']==7].value_counts())
# pd.DataFrame(X_train).to_csv('../X_train.csv',index=False)
# pd.DataFrame(X_test).to_csv('../X_test.csv',index=False)
# pd.DataFrame(y_train).to_csv('../y_train.csv',index=False)
# pd.DataFrame(y_test).to_csv('../y_test.csv',index=False)

X_train.replace(-999,None,inplace=True)
sns.boxplot(data=X_train[['Количество комнат',"Общая площадь","Всего этажей","Этаж"]],orient='h')
plt.xlabel("Общее значение",fontsize=22)
plt.ylabel("Название признаков",fontsize=22)
plt.xticks(size=22)
plt.yticks(size=22)
plt.show()
#print(data["Общая площадь"].unique())
#print(data["Этаж"].unique())
#print(data["Всего этажей"].unique())
# print(data["Серия"].unique())
# print(data[data["Серия"]=='пентхаус'].value_counts)
#
# print(data["Цена"].value_counts())
# print(data["Количество просмотров"].value_counts())
# print(data["Тип предложения"].value_counts())
# print(X_train["Серия"].value_counts())
# print(data["Отопление"].value_counts())
# print(data["Состояние"].value_counts())
# print(data["Телефон"].value_counts())
# print(data["Интернет"].value_counts())
# print(data["Санузел"].value_counts())
# print(data["Балкон"].value_counts())
# print(data["Входная дверь"].value_counts())
# print(data["Парковка"].value_counts())
# print(data["Газ"].value_counts())
# print(data["Мебель"].value_counts())
# print(data["Пол"].value_counts())
# print(data["Высота потолков"].value_counts())
# print(data["Возможность рассрочки"].value_counts())
# print(data["Возможность ипотеки"].value_counts())
# print(data["Возможность обмена"].value_counts())
# print(data["Широта"].value_counts())
# print(data["Долгота"].value_counts())
# print(data["Количество комнат"].value_counts())
# print(data["Дней с создания поста"].value_counts())
# print(data["Дней с поднятия поста"].value_counts())
# print(data["Материал строения"].value_counts())
# print(data["Год постройки"].value_counts())
# print(data["Этаж"].value_counts())
# print(data["Всего этажей"].value_counts())
# print(data["Общая площадь"].value_counts())
# print(data["Жилая площадь"].value_counts())
# print(data["Площадь кухни"].value_counts())
# print(data["решетки на окнах"].value_counts())
# print(data["сигнализация"].value_counts())
# print(data["охрана"].value_counts())
# print(data["видеонаблюдение"].value_counts())
# print(data["домофон"].value_counts())
# print(data["видеодомофон"].value_counts())
# print(data["кодовый замок"].value_counts())
# print(data["консьерж"].value_counts())
# print(data["пластиковые окна"].value_counts())
# print(data["кухня-студия"].value_counts())
# print(data["кондиционер"].value_counts())
# print(data["комнаты изолированы"].value_counts())
# print(data["кладовка"].value_counts())
# print(data["удобно под бизнес"].value_counts())
# print(data["новая сантехника"].value_counts())
# print(data["неугловая"].value_counts())
# print(data["встроенная кухня"].value_counts())
# print(data["тихий двор"].value_counts())
# print(data["срочно"].value_counts())
# print(data["торг"].value_counts())
# print(data["не торг"].value_counts())
# print(data["Длина описания"].value_counts())