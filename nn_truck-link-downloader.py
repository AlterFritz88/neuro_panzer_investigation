import requests
from lxml.html import fromstring
import os
import random as rd
from names_models import *
from get_char import *
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model



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




#c 8000 to 11026


#вроде всё выкачал

for model_page in range(7965, 8000, 1):
    print(model_page)

    url = "https://www.track-link.com/gallery/{}".format(model_page)

    r = requests.get(url)
    tree = fromstring(r.content)
    path = ' '.join(tree.findtext('.//title').split(' ')[4:]).replace(r'/', ' ').replace(r'!', ' ').replace(r'?', ' ').replace(r':', ' ').replace(r'#', ' ').replace(r'+', ' ')
    print('1. {}'.format(path))
    if len(path) < 3:      # если в шапке пусто, то скипаем, чтобы не мусорить
        continue

    code_str = [mapping_age[char] for char in path]
    padded = pad_sequences([code_str], maxlen=max_len_age, padding='post')
    list_of_pred = model_age.predict(padded)[0]
    top_age = sorted(range(len(list_of_pred)), key=lambda i: list_of_pred[i], reverse=True)[:1]
    if labels_age[top_age[0] - 1] == 'WWII':
        age = 'WWII'
        code_str = [mapping_wwii[char] for char in path]
        padded = pad_sequences([code_str], maxlen=max_len_wwii, padding='post')
        list_of_pred_nation = model_wwii.predict(padded)[0]
        top = sorted(range(len(list_of_pred_nation)), key=lambda i: list_of_pred_nation[i], reverse=True)[:4]
        nation = labels_wwii[top[0] - 1]
    else:
        age = 'Modern'
        code_str = [mapping_modern[char] for char in path]
        padded = pad_sequences([code_str], maxlen=max_len_modern, padding='post')
        list_of_pred_nation = model_modern.predict(padded)[0]
        top = sorted(range(len(list_of_pred_nation)), key=lambda i: list_of_pred_nation[i], reverse=True)[:4]
        nation = labels_modern[top[0] - 1]
    print(age, nation)
    print()




    dirName = 'truck-link/{0}/{1}/{2}'.format(age, nation, path)

    if not os.path.exists('truck-link/{0}/{1}'.format(age, nation)):
        os.mkdir('truck-link/{0}/{1}'.format(age, nation))


    if not os.path.exists(dirName):
        os.mkdir(dirName)

    try:
        for i in range(20):
            image_url = "https://www.track-link.com/gallery/images/b_{0}_{1}.jpg".format(model_page, i)
            r = requests.get(image_url)
            filename = 'truck-link/{0}/{1}/{2}/{3}.jpeg'.format(age, nation, path, i + rd.randint(10, 100000))

            if os.path.isfile(filename):
                filename = 'truck-link/{0}/{1}/{2}/{3}.jpeg'.format(age, nation, path, i + rd.randint(10, 100000))
                with open(filename, 'wb') as f:
                    f.write(r.content)
            else:

                with open(filename, 'wb') as f:
                    f.write(r.content)
    except:
        continue
