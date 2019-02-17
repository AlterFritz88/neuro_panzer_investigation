import os
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import  TfidfVectorizer, CountVectorizer
import random as rd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from get_char import *

'''
кластеризует собранные данные по папкам и удаляет пустные
важно знать на сколько видов делить!!!!
работать с бэкапом данных!!!!
'''
model = load_model('models/tank_not_tank_test.h5f')
li_dir = os.listdir(path="truck-link/sample")
print(li_dir)
li_dir_new = [i[:17] if len(i) > 15 else i for i in li_dir]
print(li_dir)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(li_dir)


'''
max_len, chars_age, mapping, labels_age = get_params_age()

padded_dirs = []

for i in li_dir:
    code_str = [mapping[char] for char in i]
    padded = pad_sequences([code_str], maxlen=max_len, padding='post')
    padded_dirs.append(padded[0])
print(padded_dirs)
'''
X = vectorizer.fit_transform(li_dir_new)

clustno = range(1, len(li_dir))

# Run MiniBatch Kmeans over the number of clusters
kmeans_more = [KMeans(n_clusters=i) for i in clustno]

# Obtain the score for each model
score = [kmeans_more[i].fit(X).score(X) for i in range(len(kmeans_more))]
kn = KneeLocator(clustno, score, curve='convex', direction='increasing')

print(kn.knee)

# Plot the models and their respective score
plt.plot(clustno, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


kmeans = KMeans(n_clusters=kn.knee).fit(X)
print(kmeans.labels_)




dick_paths = {}
for i in range(len(li_dir)):
    if kmeans.labels_[i] not in dick_paths.keys():
        dick_paths[kmeans.labels_[i]] = []
        dick_paths[kmeans.labels_[i]].append(li_dir[i])
    else:
        dick_paths[kmeans.labels_[i]].append(li_dir[i])
print(dick_paths)

for key, value in dick_paths.items():

    for i in range(len(value)):
        if i == 0:
            temp_path = 'truck-link/sample/{0}/'.format(value[0])
            print('into', temp_path)

        else:
            for item in os.listdir('truck-link/sample/{0}'.format(value[i])):
                image = cv2.imread('truck-link/sample/{0}/{1}'.format(value[i], item))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (92, 92))
                image = image.astype("float") / 255.0
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                list_of_pred = model.predict(image)[0]

                sourse = 'truck-link/sample/' + value[i] + '/' + item
                destin = temp_path + str(rd.randint(10,10000000))+ '---' +  str(list_of_pred[0]) + '.jpeg'
                if os.stat(sourse).st_size < 25000 or list_of_pred[1] > 0.4:
                    print(list_of_pred[1])
                    os.remove(sourse)
                else:
                    os.rename(sourse,  destin)
            os.rmdir('truck-link/sample/{0}'.format(value[i]))
