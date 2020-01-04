import requests
import lxml.html
from bs4 import *
from time import sleep
import random as rd
import re
import os
from transliterate import translit

from keras.preprocessing.sequence import pad_sequences
from names_models import *

from get_char import *

max_len_wwii, chars_wwii, mapping_wwii, labels_wwii = get_params_wwii()
max_len_modern, chars_modern, mapping_modern, labels_modern = get_params_modern()
max_len_age, chars_age, mapping_age, labels_age = get_params_age()

model_age = load_model('models/age_names.h5f')
model_wwii = load_model('models/wwii_names.h5f')
model_modern = load_model('models/modern_names.h5f')
model_age._make_predict_function()
model_wwii._make_predict_function()
model_modern._make_predict_function()


labels_wwII = []
with open('spisok', 'r') as file:
    for line in file:
        if len(line) < 3:
            continue
        line_no_spaces = line.replace(' ', '')
        try:
            start = int(line_no_spaces[0])
        except:
            labels_wwII.append(line_no_spaces[:-1])
            continue

labels_modern = []
with open('modern_tech', 'r') as file:
    for line in file:
        if len(line) < 3:
            continue
        line_no_spaces = line.replace(' ', '')
        try:
            start = int(line_no_spaces[0])
        except:
            labels_modern.append(line_no_spaces[:-1])
            continue


def get_title(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    title = soup.find_all('title')
    title = translit(str(title[0])[7:], "ru", reversed=True)
    a = re.search(r'\b(Karopka.ru)\b', title)
    end_point = a.start() - 3
    return title[:end_point].replace(r'/', ' ').replace(r'/', ' ').replace(r'!', ' ').replace(r'?', ' ').replace(r':', ' ').replace(r'#', ' ').replace(r'+', ' ').replace(r'"', ' ').replace(r'…', ' ').replace('_', ' ').replace('(', ' ').replace(')', ' ').replace(';', ' ')

def get_photo(url):
    photo_list = []
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    a = soup.find_all('img')
    for st in a:
        if st['src'][:14] != '/upload/resize':
            continue
        photo_list.append('https://karopka.ru' + st['src'])
    return photo_list

def define_military_civil(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    lines = soup.find_all('dd')
    classes = lines[7]
    if 'Военные' in classes:
        return 'military'
    else:
        return 'civil'




for i in range(161, 194):
    print(i)
    url = "https://karopka.ru/catalog/truck/?f=-1&p={0}".format(i)
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')  # parse content/ page source

    for a in soup.find_all('a', {'class': 'link'}, href=True):
        sleep(0.2)
        url_model = 'https://karopka.ru' + a['href']
        title_model = get_title(url_model)
        classes = define_military_civil(url_model)
        print('1. '+title_model)
        if classes == 'military':
            try:
                code_str = [mapping_age[char] for char in title_model]
                padded = pad_sequences([code_str], maxlen=max_len_age, padding='post')
                list_of_pred = model_age.predict(padded)[0]
                top_age = sorted(range(len(list_of_pred)), key=lambda i: list_of_pred[i], reverse=True)[:1]
                if labels_age[top_age[0] - 1] == 'WWII':
                    age = 'WWII'
                    code_str = [mapping_wwii[char] for char in title_model]
                    padded = pad_sequences([code_str], maxlen=max_len_wwii, padding='post')
                    list_of_pred_nation = model_wwii.predict(padded)[0]
                    top = sorted(range(len(list_of_pred_nation)), key=lambda i: list_of_pred_nation[i], reverse=True)[
                          :4]
                    nation = labels_wwii[top[0] - 1]
                else:
                    age = 'Modern'
                    code_str = [mapping_modern[char] for char in title_model]
                    padded = pad_sequences([code_str], maxlen=max_len_modern, padding='post')
                    list_of_pred_nation = model_modern.predict(padded)[0]
                    top = sorted(range(len(list_of_pred_nation)), key=lambda i: list_of_pred_nation[i], reverse=True)[
                          :4]
                    nation = labels_modern[top[0] - 1]
                print(age, nation)
                print()
            except:
                continue

            dirName = 'truck-link/{0}/{1}/{2}'.format(age, nation, title_model)

            if not os.path.exists('truck-link/{0}/{1}'.format(age, nation)):
                os.mkdir('truck-link/{0}/{1}'.format(age, nation))

            if not os.path.exists(dirName):
                os.mkdir(dirName)

            try:
                photo_list = get_photo(url_model)

                for photo in photo_list:
                    r = requests.get(photo)
                    filename = 'truck-link/{0}/{1}/{2}/{3}-{4}.jpeg'.format(age, nation, title_model, 'karopka',
                                                                            i + rd.randint(10, 10000000))
                    with open(filename, 'wb') as f:
                        f.write(r.content)
            except:
                continue
        else:
            dirName = 'truck-link/{0}/{1}/{2}'.format('civil', 'civil', title_model)



            if not os.path.exists(dirName):
                os.mkdir(dirName)

            try:
                photo_list = get_photo(url_model)

                for photo in photo_list:
                    r = requests.get(photo)
                    filename = 'truck-link/{0}/{1}/{2}/{3}-{4}.jpeg'.format('civil', 'civil', title_model, 'karopka',
                                                                            i + rd.randint(10, 10000000))
                    with open(filename, 'wb') as f:
                        f.write(r.content)
            except:
                continue