"""
# Table of contents
### 1 Preparatory steps
&emsp;**1.1 Importing libraries**<br>
&emsp;**1.2 Importing scraped data**<br>
### 2 Extracting info from *title*
&emsp;**2.1 House type**<br>
&emsp;**2.2 Number of rooms**<br>
### 3 Extraction info from *address*
&emsp;**3.1 Extraction of city**<br>
&emsp;**3.2 Extraction of district**<br>
### 4 Formatting prices
&emsp;**4.1 Removing prices in soms**<br>
&emsp;**4.2 Reformatting prices in dollars**<br>
### 5 Formatting post and update dates
&emsp;**5.1 Defining function**<br>
&emsp;**5.2 Applying function**<br>
### 6 Extracting info from *house_type*
&emsp;**6.1 Extraction of material type**<br>
&emsp;**6.2 Extraction of construction year**<br>
### 7 Formatting *floor*
### 8 Formatting *area*
### 9 Formatting *communication/infrastructure*
### 10 Formatting *security*
### 11 Formatting *other*
### 12 Formatting *ceiling_height*
### 13 Extracting info from *description*
### 14 Reformatting latitude and longitude
### 15 Dropping unnecessary columns
### 16 Minor preprocessing before saving data

---

# 1 Preparatory steps

### 1.1 Importing libraries
"""
import time
import random
from glob import glob
import re
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from selenium import webdriver
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.utils import ChromeType
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = lambda x: "%.5f" % x
pd.options.display.max_columns = None
pd.options.display.max_rows = None


csv_files = glob('.\data\*.csv')
full_data = pd.DataFrame()
for csv_file in csv_files:
    print(csv_file)
    data = pd.read_csv(csv_file)
    print(data.head())
    data['file_number'] = int(re.search("[0-9]+", csv_file).group(0))
    full_data = pd.concat([full_data, data], ignore_index=True)


print(full_data.shape)

"""Let's rename "Дом" to "house_type" since in the future steps we will create other variable called "Дом":"""

full_data = full_data.rename(columns={'Дом':'house_type'})

"""---

# 2 Extracting info from *title*

### 2.1 House type
"""

full_data.head()

full_data['Тип недвижимости'] = "Не указано"

full_data.loc[full_data.title.str.contains('кв'), 'Тип недвижимости'] = "Квартира"

house_types = ['Дом', 'Участок', 'Гараж', 'Коммерческая недвижимость', 'Дача']

for house_type in house_types:
    full_data.loc[full_data.title.str.contains(house_type), "Тип недвижимости"] = house_type

full_data.head()

full_data['Тип недвижимости'].value_counts()

"""### 2.2 Number of rooms"""

full_data.head()

def extract_number_of_rooms(row):
    if row['Тип недвижимости']=="Квартира":
        number_of_rooms = row.title.split("-")[0].strip()
    elif row["Тип недвижимости"]=='Дом' or row["Тип недвижимости"]=='Дача':
        number_of_rooms = row.title.split(",")[1].split("-")[0].strip()
    try:
        number_of_rooms = float(re.search('[0-9]+', number_of_rooms).group(0))
    except:
        number_of_rooms = np.nan
    return number_of_rooms

full_data['Количество комнат'] = full_data.apply(extract_number_of_rooms, axis=1)

full_data[['title', 'Количество комнат']].head()

"""---

# 3 Extraction info from *address*
"""

full_data.address.head()

# """### 3.1 Extraction of city"""
#
# full_data['Город'] = full_data.address.apply(lambda x: x.split(",")[0].strip())
#
# full_data['Город'].value_counts()[0:10]
#
# """There are some numbers in city names, so let's deal with such cases by substituting all numbers to whitespaces:"""
#
# full_data['Город'] = full_data['Город'].apply(lambda x: re.sub("\d+", "", x).strip())
#
# """Also such city names that starts from "c." can be considered as villages, so let's extract this info as well:"""
#
# full_data['Село'] = 0
# full_data.loc[full_data['Город'].apply(lambda x: x[:2])=='с.', 'Село'] = 1
#
# full_data['Село'].value_counts(normalize=True)
#
# """### 3.2 Extraction of district
#
# Here it will be hard to account for all cases, since there are such ones where district is not specified, however address is specified. So, we will still extract this info and assume that model will ignore those cases where district is incorrect since there are not so many cases like this:
# """
#
# full_data['Район'] = full_data.address.apply(lambda x: x.split(",")[1] if "," in x else np.nan)
#
# full_data['Район'].value_counts(dropna=False)
#
"""---

# 4 Formatting prices

### 4.1 Removing prices in soms

Let's delete all prices in soms and leave only those that are in dollars:
"""

full_data = full_data.drop(columns=['price_som', 'som_per_m2'])

full_data.head()

"""### 4.2 Reformatting prices in dollars"""

full_data.price_dollar = full_data.price_dollar.apply(lambda x: float(x.replace("$", "").replace(" ", "")))

full_data.price_dollar.head()

full_data.dollar_per_m2.str.contains('/м2').mean()

"""All prices per m2 are actually in m2, so we can remove everything except actual value:"""

full_data.dollar_per_m2 = full_data.dollar_per_m2.apply(lambda x: float(x.split("/")[0].replace("$", "").replace(" ", ""))
                                                        if pd.isnull(x)==False else np.nan)

full_data.dollar_per_m2.head()

"""---

# 5 Formatting post and update dates

### 5.1 Defining function
"""

full_data[['post_date', 'update_date']].head()

def format_date(raw_value):
    formatted_value = np.nan
    if "минут" in raw_value:
        formatted_value = float(re.search("[0-9]+", raw_value).group(0)) / 24 / 60
    elif "час" in raw_value:
        formatted_value = float(re.search("[0-9]+", raw_value).group(0)) / 24
    elif "день" in raw_value or "дней" in raw_value or "дня" in raw_value:
        formatted_value = float(re.search("[0-9]+", raw_value).group(0))
    elif "недел" in raw_value:
        formatted_value = float(re.search("[0-9]+", raw_value).group(0)) * 7
    elif "месяц" in raw_value:
        formatted_value = float(re.search("[0-9]+", raw_value).group(0)) * 30
    elif "год" in raw_value:
        formatted_value = float(re.search("[0-9]+", raw_value).group(0)) * 365
    return formatted_value

"""### 5.2 Applying function"""

full_data['Дней с создания поста'] = full_data.post_date.apply(format_date)

full_data[['post_date','Дней с создания поста']].head()

full_data['Дней с поднятия поста'] = full_data.update_date.apply(lambda x: format_date(str(x)[8:]))

full_data[['update_date','Дней с поднятия поста']].head()

full_data = full_data.drop(columns=['post_date', 'update_date'])

full_data.head()

"""---

# 6 Extracting info from *house_type*

### 6.1 Extraction of material type
"""

full_data['Материал строения'] = full_data.house_type.apply(lambda x: x.split(",")[0]
                                                         if pd.isnull(x)==False else np.nan)

full_data['Материал строения'].value_counts(dropna=False)

"""### 6.2 Extraction of construction year"""

def get_house_year(raw_value):
    if pd.isnull(raw_value)==False and re.search('[0-9]+', raw_value):
        year = int(re.search('[0-9]+', raw_value).group(0))
    else:
        year = np.nan
    return year

full_data['Год постройки'] = full_data.house_type.apply(get_house_year)

full_data.loc[full_data['Год постройки']<1900, 'Год постройки'] = np.nan

full_data['Год постройки'].value_counts(normalize=True, dropna=False).head()

"""---

# 7 Formatting *floor*
"""

full_data[full_data['Этаж'].fillna('MISSING').str.contains('подвал')]

def format_floor(value):
    if pd.isnull(value)==False and "цоколь" in value:
        floor = 0
        max_floors = int(value.split(" ")[-2])
    elif pd.isnull(value)==False and "подвал" in value:
        floor = -1
        max_floors = int(value.split(" ")[-2])
    elif pd.isnull(value)==False:
        floor = int(value.split(" ")[0])
        max_floors = int(value.split(" ")[-1])
    else:
        floor = np.nan
        max_floors = np.nan
    return pd.Series([floor, max_floors])

full_data[['floor', 'max_floors']] = full_data['Этаж'].apply(format_floor)

full_data[['floor', 'max_floors', 'Этаж']]

full_data = full_data.drop(columns='Этаж')

full_data = full_data.rename(columns={'floor':'Этаж', 'max_floors':'Всего этажей'})

"""---

# 8 Formatting *area*
"""

full_data['Площадь'].value_counts(dropna=False)

def extract_info_from_area(raw_value):
    area = np.nan
    living_area = np.nan
    kitchen_area = np.nan
    if pd.isnull(raw_value)==False:
        area = float(raw_value.split(",")[0].split(" ")[0])
        if "жилая" in raw_value:
            living_area = [float(area.strip().split(" ")[1]) for area in raw_value.split(",") if "жилая" in area][0]
        if "кухня" in raw_value:
            kitchen_area = [float(area.strip().split(" ")[1]) for area in raw_value.split(",") if "кухня" in area][0]
    return pd.Series([area, living_area, kitchen_area])

full_data[['Общая площадь', 'Жилая площадь', 'Площадь кухни']] = full_data['Площадь'].apply(extract_info_from_area)

full_data[['Общая площадь', 'Жилая площадь', 'Площадь кухни']]

full_data['Площадь участка'].value_counts(dropna=False)[0:20]

full_data['Площадь участка'] = full_data['Площадь участка'].apply(lambda x: float(x.split(" ")[0])
                                                                  if pd.isnull(x)==False else np.nan)

full_data['Площадь участка']

"""---

# 9 Formatting *communication/infrastructure*
"""


full_data['Коммуникации'][full_data['Коммуникации'].notnull()].head()

communications_list = ['свет', 'интернет', 'вода', 'канализация', 'газ', 'отопление', 'телефон']

for communication in communications_list:
    full_data['участок_' + communication] = 0
    full_data.loc[full_data['Коммуникации'].notnull(), 'участок_' + communication] = full_data['Коммуникации'].dropna().\
    apply(lambda x: 1 if communication in x else 0)

full_data.loc[full_data['Коммуникации'].notnull(), full_data.columns.str.contains("Коммуникации|участок")].head()

"""---

# 10 Formatting *security*
"""

full_data.head()

security_list = ['решетки на окнах', 'сигнализация', 'охрана', 'видеонаблюдение',
                 'домофон', 'видеодомофон', 'кодовый замок', 'консьерж']

for security in security_list:
    full_data[security] = 0
    full_data.loc[full_data['Безопасность'].notnull(), security] = full_data['Безопасность'].dropna().\
    apply(lambda x: 1 if security in x else 0)

full_data.loc[:, ['Безопасность'] + full_data.columns[full_data.columns.isin(security_list)].tolist()][100:110]

"""---

# 11 Formatting *other*
"""

full_data.head()

other_list = ['пластиковые окна', 'кухня-студия', 'кондиционер', 'комнаты изолированы', 'кладовка', 'удобно под бизнес',
              'улучшенная', 'новая сантехника', 'неугловая', 'встроенная кухня', 'тихий двор', 'бассейн', 'хозпостройки',
              'сауна', 'веранда', 'баня', 'сад', 'навес', 'гараж', '3 фазы', 'фундамент', 'ж/д тупик', 'удобно под коммерцию',
              'времянка', 'удобный въезд', 'возможна аренда', 'огорожен', 'госакт', 'дом', 'ровный', 'все документы']

for other_category in other_list:
    full_data[other_category] = 0
    full_data.loc[full_data['Разное'].notnull(), other_category] = full_data['Разное'].dropna().\
    apply(lambda x: 1 if other_category in x else 0)

full_data.loc[:, ['Разное'] + full_data.columns[full_data.columns.isin(other_list)].tolist()][100:110]

"""---

# 12 Formatting *ceiling_height*
"""

full_data['Высота потолков'].value_counts(dropna=False)[0:20]

full_data['Высота потолков'] = full_data['Высота потолков'].apply(lambda x: float(x.split(" ")[0])
                                                                  if pd.isnull(x)==False else np.nan)

full_data['Высота потолков'].head()

"""---

# 13 Extracting info from *description*
"""

full_data.description[0]

full_data['urgent'] = 0
full_data.loc[full_data.description.fillna("MISSING").str.lower().str.contains('срочно'), 'urgent'] = 1

full_data.urgent.value_counts()

full_data['торг'] = 0
full_data.loc[(full_data.description.fillna("MISSING").str.lower().str.contains('торг')) &
              (~full_data.description.fillna("MISSING").str.lower().str.contains('не торг|торг не')), 'торг'] = 1

full_data['торг'].value_counts()

full_data['не торг'] = 0
full_data.loc[full_data.description.fillna("MISSING").str.lower().str.contains('не торг|торг не'), 'не торг'] = 1

full_data['не торг'].value_counts()

full_data['Длина описания'] = full_data.description.astype(str).apply(len)

full_data['Длина описания'].describe()

"""---

# 14 Reformatting latitude and longitude
"""

full_data[['latitude', 'longitude']].head()

full_data.latitude = full_data.latitude.apply(lambda x: float(re.search("[\d\.]+", x).group(0))
                                              if pd.isnull(x)==False else x)

full_data.latitude.head()

full_data.longitude = full_data.longitude.apply(lambda x: float(re.search("[\d\.]+", x).group(0))
                                                if pd.isnull(x)==False else x)

full_data.longitude.head()

"""---

# 15 Dropping unnecessary columns
"""

# full_data.head()
#
columns_to_drop = ['house_type']
#
full_data = full_data.drop(columns=columns_to_drop)

"""---

# 16 Minor preprocessing before saving data

Let's rename columns using names in Russian:
"""

full_data = full_data.rename(columns={
    'title':'Название',
    'price_dollar':"Цена",
    'dollar_per_m2':"Цена за м2",
    'views': 'Количество просмотров',
    'latitude':"Широта",
    "longitude":"Долгота"})

"""Areas for "Участок" are in different scale, which is "сотка", and since 1 "сотка" is 100 m2, we need to multiply all the values by 100:"""

full_data.loc[full_data['Тип недвижимости']=="Участок",
              "Общая площадь"] = full_data.loc[full_data['Тип недвижимости']=="Участок", "Площадь участка"] * 100

full_data = full_data.drop(columns="Площадь участка")

"""Now let's recalculate price for m2 for category "Участок", because before it was dollar for 1 "сотка":"""

full_data.loc[full_data['Тип недвижимости']=="Участок","Цена за м2"] = \
full_data.loc[full_data['Тип недвижимости']=="Участок", "Цена"] / full_data.loc[full_data['Тип недвижимости']=="Участок", "Общая площадь"]

"""Let's also create variable, which will reflect offer number (it might give some additional information for the model and also will be useful for DASH APP, which will be final outcome of all work):"""

full_data['offer_number'] = full_data.index

full_data.to_csv(f'.\data\interim_data.csv', index=False)

print(full_data.shape)
print(full_data.columns)