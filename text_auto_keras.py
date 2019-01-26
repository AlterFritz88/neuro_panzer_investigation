from sklearn.model_selection import train_test_split
from transliterate import translit
from autokeras import TextClassifier
import os
import keras
from keras_preprocessing import image

import torch
print(torch.rand(1, device="cpu"))
import os



if __name__ == '__main__':

    labels = []
    data_dict = {}
    data = []
    label = []
    with open('modern_tech', 'r') as file:
        for line in file:
            if len(line) < 3:
                continue
            line_no_spaces = line.replace(' ', '')
            try:
                start = int(line_no_spaces[0])
            except:
                labels.append(line_no_spaces[:-1])
                continue

            for i in range(len(line)):
                if line[i] == ' ':
                    continue
                if line[i] == '.':
                    line = line[i+2:]
                    break

            data_dict[line[:-1]] = len(labels)
            data.append(translit(u"{}".format(line[:-1]), "ru", reversed=True))
            label.append(len(labels))
    # добавляем подкрепление
    '''
    
    for dir in labels:
        if dir in os.listdir('truck-link/Modern'):
            for text in os.listdir('truck-link/Modern/{0}'.format(dir)):
                data.append(text)
                label.append(labels.index(dir) + 1)
    '''




    trainX, testX, trainY, testY = train_test_split(data, label, test_size = 0.3, random_state = 42)




    clf = TextClassifier(verbose=True)
    clf.fit(x=trainX, y=trainY, time_limit=3 * 60 * 60)
    clf.final_fit(trainX, trainY, testX, testY, retrain=True)
    y = clf.evaluate(testX, testY)
    print(y * 100)