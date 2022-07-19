# 1 Preparatory steps

### 1.1 Importing libraries


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

"""### 1.2 Initializing Google Chrome browser instance """

link="https://www.house.kg/kupit?page=1"

options = webdriver.ChromeOptions()

options.add_argument('--ignore-certificate-errors')

options.add_argument("--incognito")

driver = webdriver.Chrome(ChromeDriverManager().install(),options=options)

driver.get(link)

"""---
# 2 Testing navigation buttons
### 2.1 Clicking house links and back button
"""

house_links = driver.find_elements_by_class_name("listing")

len(house_links)

house_links[0].find_element_by_class_name('title').find_element_by_tag_name('a').click()

driver.back()

"""### 2.2 Navigating between pages"""

navigation_buttons = driver.find_elements_by_class_name('page-link')

navigation_buttons[8].text.strip()

next_page = [button for button in navigation_buttons if button.text.strip()=="»\nВперед"]

next_page[0].click()

"""---
# 3 Scraping one house information
"""

house_links = driver.find_elements_by_class_name("listing")

house_links[0].find_element_by_class_name('title').find_element_by_tag_name('a').click()

page_soup = BeautifulSoup(driver.page_source, "lxml")

"""### 3.1 General information"""

# general_info = page_soup.find("div", attrs={'class':'details-header'})

# title = general_info.find('h1').text.strip()

# address = general_info.find('div', attrs={'class':'adress'}).text.strip()

# price_dollar = general_info.find('div', attrs={'class':'sep main'}).find('div', attrs={'class':'price-dollar'}).text.strip()

# price_som = general_info.find('div', attrs={'class':'sep main'}).find('div', attrs={'class':'price-som'}).text.strip()

# dollar_per_m2 = general_info.find('div', attrs={'class':'sep addit'}).find('div', attrs={'class':'price-dollar'}).text.strip()

# som_per_m2 = general_info.find('div', attrs={'class':'sep addit'}).find('div', attrs={'class':'price-som'}).text.strip()

# post_date = general_info.find('span', attrs={'class':'added-span'}).text.strip()

# upped_date = general_info.find('span', attrs={'class':'upped-span'}).text.strip()

# views = general_info.find('span', attrs={'class':'view-count'}).text.strip()

# """### 3.2 Details"""
# details_info = page_soup.find('div', attrs={'class':'details-main'})

# details_list = details_info.find_all('div', attrs={'class':'info-row'})

# labels_list = [detail.find('div', attrs={'class':'label'}).text.strip() for detail in details_list]

# values_list = [detail.find('div', attrs={'class':'info'}).text.strip() for detail in details_list]

# {label:value for (label, value) in zip(labels_list, values_list)}

"""### 3.3 Description"""

#description = page_soup.find('div', attrs={'class':'description'}).text

"""---
# 4 Automation
### 4.1 Defining function for scraping one house information
"""

def scrape_one_house_info(page_soup):
    house_info = dict()

    general_info = page_soup.find("div", attrs={'class':'details-header'})
    general_info_dict = scrape_general_info(general_info)
    house_info.update(general_info_dict)

    details_info = page_soup.find('div', attrs={'class':'details-main'})
    details_dict = scrape_details(details_info)
    house_info.update(details_dict)

    if 'Описание от продавца' in str(page_soup):
        description = page_soup.find('div', attrs={'class':'description'}).text
        house_info.update({'description' : description})

    coordinates_dict = scrape_coordinates(page_soup)
    house_info.update(coordinates_dict)

    current_url = driver.current_url
    house_info.update({'link' : current_url})
    return house_info


def scrape_general_info(general_info):
    general_info_dict = dict(
        title = general_info.find('h1').text.strip(),
        address = general_info.find('div', attrs={'class':'adress'}).text.strip(),
        price_dollar = general_info.find('div', attrs={'class':'sep main'}).find('div', attrs={'class':'price-dollar'}).text.strip(),
        price_som = general_info.find('div', attrs={'class':'sep main'}).find('div', attrs={'class':'price-som'}).text.strip(),
        post_date = general_info.find('span', attrs={'class':'added-span'}).text.strip(),
        views = general_info.find('span', attrs={'class':'view-count'}).text.strip()
    )

    if 'sep addit' in str(general_info):
        sep_addit_dict = dict(
            dollar_per_m2 = general_info.find('div', attrs={'class':'sep addit'}).find('div', attrs={'class':'price-dollar'}).text.strip(),
            som_per_m2 = general_info.find('div', attrs={'class':'sep addit'}).find('div', attrs={'class':'price-som'}).text.strip()
        )
        general_info_dict.update(sep_addit_dict)

    if 'upped-span' in str(general_info):
        update_date_dict = dict(update_date = general_info.find('span', attrs={'class':'upped-span'}).text.strip())
        general_info_dict.update(update_date_dict)
    return general_info_dict


def scrape_details(details_info):
    details_list = details_info.find_all('div', attrs={'class':'info-row'})
    labels_list = [detail.find('div', attrs={'class':'label'}).text.strip() for detail in details_list]
    values_list = [detail.find('div', attrs={'class':'info'}).text.strip() for detail in details_list]
    details_dict = {label:value for (label, value) in zip(labels_list, values_list)}
    return details_dict

def scrape_coordinates(page_soup):
    latitude = np.nan
    longitude = np.nan
    if "data-lat" in str(page_soup):
        latitude = str(page_soup).split('data-lat=')[1].split(" ")[0]
    if "data-lon" in str(page_soup):
        longitude = str(page_soup).split('data-lon=')[1].split(" ")[0]
    return dict(latitude = latitude, longitude = longitude)

page_soup = BeautifulSoup(driver.page_source, "lxml")

one_job_info = scrape_one_house_info(page_soup)

"""### 4.2 Initializing empty dataframe"""

columns_for_df = [
    'title',
    'address',
    'price_dollar',
    'price_som',
    'dollar_per_m2',
    'som_per_m2',
    'post_date',
    'update_date',
    'views',
    'Тип предложения',
    'Дом',
    'Этаж',
    'Кол-во этажей',
    'Площадь',
    'Площадь участка',
    'Серия',
    'Отопление',
    'Состояние',
    'Телефон',
    'Интернет',
    'Санузел',
    'Балкон',
    'Входная дверь',
    'Парковка',
    'Коммуникации',
    'Местоположение',
    'Канализация',
    'Питьевая вода',
    'Электричество',
    'Газ',
    'Поливная вода',
    'Безопасность',
    'Мебель',
    'Пол',
    'Высота потолков',
    'Разное',
    'Возможность рассрочки',
    'Возможность ипотеки',
    'Возможность обмена',
    'description',
    'latitude',
    'longitude',
    'link'
]

houses_df = pd.DataFrame(columns=columns_for_df)

"""### 4.3 Actual scraping"""

def launch_browser(start_page_number):
    link=f"https://www.house.kg/kupit?region=1&town=2&sort_by=upped_at%20desc&page={start_page_number}"
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument("--incognito")
    driver = webdriver.Chrome(ChromeDriverManager().install(),options=options)
    driver.get(link)
    return driver

def scrape_one_page(driver, houses_df):
    house_links = driver.find_elements_by_class_name("listing")
    for house_link_number in range(len(house_links)):
        click_house_link(driver, house_link_number)
        try:
            time.sleep(10)
            houses_df = scrape_house_info_and_append(driver, houses_df)
        except:
            try:
                time.sleep(5)
                houses_df = scrape_house_info_and_append(driver, houses_df)
            except:
                print(f"Error at page {page_number}, house {house_link_number+1}")
        return_to_main_page()
        time.sleep(random.randint(1, 6))
    return houses_df

def click_house_link(driver, house_link_number):
    house_links = driver.find_elements_by_class_name("listing")
    house_links[house_link_number].find_element_by_class_name('title').find_element_by_tag_name('a').click()

def scrape_house_info_and_append(driver, houses_df):
    page_soup = BeautifulSoup(driver.page_source, "lxml")
    one_house_info = scrape_one_house_info(page_soup)
    houses_df = houses_df.append(one_house_info, ignore_index=True)
    return houses_df

def return_to_main_page():
    driver.back()

def navigate_to_next_page(driver):
    navigation_buttons = driver.find_elements_by_class_name('page-link')
    next_page_button = [button for button in navigation_buttons if button.text.strip()=="»\nВперед"]
    next_page_button[0].click()

def save_scraped_data(houses_df, page_number):
    #houses_df = houses_df[houses_df["address"].str.contains("Бишкек")==True]
    houses_df.to_csv ( f'houses_df_{page_number}.csv', index=False )
    print(f"Everything till page {page_number} (inclusive) has been saved")

start_page_number =1
end_page_number = 51

houses_df = pd.DataFrame(columns=columns_for_df)
driver = launch_browser(start_page_number)

for page_number in tqdm(np.arange(start_page_number, end_page_number), position=0, leave=True):
    try:
        houses_df = scrape_one_page(driver, houses_df)
    except:
        try:
            time.sleep(3)
            houses_df = scrape_one_page(driver, houses_df)
        except:
            print(f"Page {page_number} couldn't be scraped")

    navigate_to_next_page(driver)

    #if page_number % 30 == 0:
     #   save_scraped_data(houses_df, page_number)
      #  houses_df = pd.DataFrame(columns=columns_for_df)

save_scraped_data(houses_df, page_number)