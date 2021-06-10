import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
pd.options.display.float_format = lambda x: "%.5f" % x
pd.options.display.max_columns = None
pd.options.display.max_rows = None

"""03. Handling incorrect cases.
# Table of contents
### 1 Preparatory steps
&emsp;**1.1 Importing libraries**<br>
&emsp;**1.2 Importing formatted scraped data**<br>
### 2 Dropping duplicates
### 3 Handling incorrect prices
### 4 Handling incorrect areas
&emsp;**4.1 Handling too low areas**<br>
&emsp;**4.2 Handling too big areas**<br>
### 5 Handling incorrect price for m2
&emsp;**5.1 Handling too low prices for m2**<br>
&emsp;**5.2 Handling too big prices for m2**<br>
### 6 Handling incorrect rooms number
### 7 Saving data

---

# 1 Preparatory steps

"""

full_data = pd.read_csv('..\data\interim_data.csv')

full_data.head()

full_data.shape

"""---

# 2 Dropping duplicates
"""

print(full_data[["Название", "address"]].duplicated().mean())

"""21,5 % of offers are duplicates, so let's remove them from dataset:"""

full_data = full_data.drop_duplicates(subset=["Название", 'address'])

full_data.shape

"""---

# 3 Handling incorrect prices

Actually once price is smaller than 1000 dollars it stops look realistic - only garages can worth such amount, so let's find all the cases where estate is not garage and has price smaller than 1000:
"""

incorrect_offers_indices = full_data[(full_data['Цена']<=1000) & (full_data['Тип недвижимости']!="Гараж")].index

full_data.loc[incorrect_offers_indices]

"""Indeed, if we look through such offers we can notice that they are not realistic at all, so let's drop them:"""

full_data = full_data.drop(index=incorrect_offers_indices)

full_data.shape

"""---

# 4 Handling incorrect areas

### 4.1 Handling too low areas

Let's handle incorrect areas, to be exact, let's deal with such cases where estate is house or flat, however the area is smaller than 10 m2, which doesn't look realistic at all:
"""

incorrect_offers_indices = full_data[(full_data['Общая площадь']<10) &
                                     (full_data['Тип недвижимости'].isin(['Квартира', 'Дом']))].index

full_data.loc[incorrect_offers_indices, :]

"""As wee see all these cases look as wrong ones, so we can safely drop them:"""

full_data = full_data.drop(index=incorrect_offers_indices)

full_data.shape

"""### 4.2 Handling too big areas"""

incorrect_offers_indices = full_data[(full_data['Общая площадь']>1000) & (full_data['Тип недвижимости']=='Квартира')].index

full_data.loc[incorrect_offers_indices, :]

full_data = full_data.drop(index=incorrect_offers_indices)

full_data.shape

"""Let's drop such observations where area is bigger than 1 billion of m2. Why 1 billion? Because there are some offers that sell huge areas of 50 mln m2 and look quite realistic and correct, but having area more than 1 billion is already not realistic at all:"""

incorrect_offers_indices = full_data[full_data['Общая площадь']>100000000].index

full_data.loc[incorrect_offers_indices, :]

"""Yeah, as we see this observation looks totally incorrect, so let's drop it:"""

full_data = full_data.drop(index=incorrect_offers_indices)

full_data.shape

"""---

# 5 Handling incorrect price for m2

### 5.1 Handling too low prices for m2

In case of price for m2, we need to take into consideration several factors:
* row land ("Участок") might have very low price for m2
* real estate located outside of Bishkek might have very low price for m2
* cases where price for m2 is very low, however actual price is high, are quite realistic

That's why let's consider such observations which are not "Участок", located in Bishkek, have price below 50000 dollars and very low price for m2:
"""

incorrect_offers_indices = full_data[(full_data['Тип недвижимости']!="Участок") &
                                     (full_data['Цена']<50000) &
                                     (full_data['Цена за м2']<20)].index

full_data.loc[incorrect_offers_indices, :]

"""All of these offers don't look realistic, so let's remove them as well:"""

full_data = full_data.drop(index=incorrect_offers_indices)

full_data.shape

"""Also let's remove cases where price for m2 is 0:"""

incorrect_offers_indices = full_data[full_data['Цена за м2']==0].index

full_data.loc[incorrect_offers_indices, :]

full_data = full_data.drop(index=incorrect_offers_indices)

full_data.shape

"""### 5.2 Handling too big prices for m2

Let's remove all the cases where price for m2 is above 10000:
"""

incorrect_offers_indices = full_data[full_data['Цена за м2']>10000].index

full_data.loc[incorrect_offers_indices, :]

"""Again, all the cases above look completely wrong, so it doesn't make sense to keep them in dataset:"""

full_data = full_data.drop(index=incorrect_offers_indices)

full_data.shape

"""---

# 6 Handling incorrect rooms number
"""

full_data[full_data['Количество комнат']>6][['Название', 'Цена', 'Количество комнат']]

"""All such cases don't look correct at all, so let's fill them with NAs:"""

full_data.loc[full_data['Количество комнат']>6, 'Количество комнат'] = np.nan

"""---
"""

full_data.head()

full_data.shape

"""
# Table of contents
### 1 Preparatory steps
&emsp;**1.1 Importing libraries**<br>
&emsp;**1.2 Importing processed scraped data**<br>
### 2 Dropping unnecessary columns
### 3 Preparation for correlation analysis
&emsp;**3.1 Division on features and target**<br>
&emsp;**3.2 Dummification of categorical features**<br>
### 4 Correlation analysis

# 2 Dropping unnecessary columns

Columns that are going to be dropped:
* "Название", address and description since we already extracted all information from them
* "Цена за м2" because it will lead to data leakage during modelling process (model will just multiply area by price for m2 to predict actual price)
"""
full_data.loc[full_data['Высота потолков']>6, 'Высота потолков']=np.nan
full_data.loc[full_data['Общая площадь']>=400, 'Общая площадь']=np.nan
full_data.loc[full_data['Этаж']<0, 'Этаж']=np.nan
full_data=full_data[full_data["Этаж"]!=0]
full_data=full_data[full_data["Цена"]<1000000]
full_data.drop(full_data.index[full_data['Год постройки'] == 2023], inplace = True)
full_data.drop(full_data.index[full_data['Год постройки'] == 2024], inplace = True)
full_data.drop(full_data.index[full_data['Год постройки'] == 2025], inplace = True)
full_data.drop(full_data.index[full_data['Год постройки'] == 2026], inplace = True)
full_data=full_data[full_data["Всего этажей"]!=1]
full_data.loc[full_data['Всего этажей']<0, 'Всего этажей']=np.nan

full_data.loc[full_data['Этаж']>16, 'Этаж']=np.nan

full_data.loc[full_data['Всего этажей']>20, 'Всего этажей']=np.nan
full_data.loc[full_data['Высота потолков']>5, 'Высота потолков']=np.nan

full_data.rename(columns={'urgent':'срочно'},inplace=True)
full_data = full_data[full_data["Тип недвижимости"]=="Квартира"]
full_data=full_data[["Общая площадь","Цена","Долгота","Состояние","Широта","Количество комнат","Количество просмотров","Всего этажей","Серия","Этаж","Длина описания","Год постройки","Высота потолков","Мебель","Возможность обмена","Отопление","Тип предложения","Санузел","кондиционер"]]
#full_data = full_data.drop(columns=['Название',"Кол-во этажей","Тип строения","Местоположение","все документы","file_number","госакт","удобно под коммерцию","возможна аренда","удобный въезд","фундамент","Канализация","Электричество","Площадь","улучшенная","Разное","Коммуникации","Питьевая вода","Безопасность",'address', 'description', 'Цена за м2','участок_свет',"участок_интернет","участок_вода",'бассейн',"участок_канализация","участок_газ","участок_отопление","участок_телефон","Поливная вода","link","Тип недвижимости","хозпостройки","сауна","веранда","баня","сад","навес","гараж","3 фазы","ж/д тупик","времянка","дом","ровный","огорожен","offer_number"])
print(full_data.shape)
columns_variation = full_data.apply(lambda x: x.value_counts(normalize=True, dropna=False).max()).sort_values(ascending=False)
print(columns_variation[0:20])
small_variance= columns_variation[columns_variation>=0.9].index
#full_data = full_data.drop(columns=small_variance)
print(full_data["кондиционер"].value_counts())
print(full_data.shape)
full_data=full_data[['Общая площадь',"Цена","Долгота","Состояние","Широта","Количество комнат","Всего этажей","Серия","Этаж","Год постройки","Санузел","Мебель","Отопление","Тип предложения"]]
def fill_missing_values(data):
    categorical_columns = data.select_dtypes(object).columns.tolist()
    data[categorical_columns] = data[categorical_columns].fillna("MISSING")

    numerical_columns = data.select_dtypes('number').columns.tolist()
    data[numerical_columns] = data[numerical_columns].fillna(-999)
    return data
full_data=fill_missing_values(full_data)
full_data.to_csv('..\data\old_processed_data.csv', index=False)
import dill
from category_encoders import CountEncoder
categorical_columns = full_data.select_dtypes(object).columns
ce=CountEncoder(cols=categorical_columns,min_group_size=1,handle_unknown=0)
full_data=ce.fit_transform(full_data)
full_data=fill_missing_values(full_data)
full_data.to_csv('..\data\processed_data.csv',index=False)


