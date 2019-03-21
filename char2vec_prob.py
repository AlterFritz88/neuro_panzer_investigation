from transliterate import translit


from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout, GlobalMaxPooling1D, Conv1D, Embedding, GlobalMaxPool1D, SeparableConv1D, CuDNNLSTM, Bidirectional, LSTM, BatchNormalization, MaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.regularizers import l2, l1, l1_l2

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
import chars2vec
import sklearn.decomposition
import matplotlib.pyplot as plt

import re

def has_cyrillic(text):
    return bool(re.search('[а-яА-Я]', text))

train = 0  # 1 = modern    0 = wwii

if train == 0:
    path = 'spisok'
    model_name = 'wwii'
else:
    path = 'modern_tech'
    model_name = 'modern'

labels = []
data_dict = {}
data = []
label = []

with open(path, 'r') as file:
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

        if train == 1:
            if len(labels) != 1 and len(labels) != 8 and len(labels) != 49 and len(labels) != 10:
                sent = line[:-1].split(' ')
                sent_new = []
                for t in sent:
                    if has_cyrillic(t) == False:
                        sent_new.append(t)
                sent_st = ' '.join(sent_new)
                if len(sent_st) < 2:
                    continue
                data_dict[sent_st] = len(labels)
                data.append(sent_st)
                label.append(len(labels))
                print(labels[len(labels)-1])
                continue
            data_dict[line[:-1]] = len(labels)
            data.append(translit(u"{}".format(line[:-1]), "ru", reversed=True))
            # data.append(line[:-1])
            label.append(len(labels))
            print(labels[len(labels)-1])
        #if len(labels) != 1 and len(labels) != 8 and len(labels) != 49 and len(labels) != 10:
        if train == 0:
            if len(labels) != 2 and len(labels) != 7:
                sent = line[:-1].split(' ')
                sent_new = []
                for t in sent:
                    if has_cyrillic(t) == False:
                        sent_new.append(t)
                sent_st = ' '.join(sent_new)
                if len(sent_st) < 2:
                    continue
                data_dict[sent_st] = len(labels)
                data.append(sent_st)
                label.append(len(labels))
                continue


            data_dict[line[:-1]] = len(labels)
            data.append(translit(u"{}".format(line[:-1]), "ru", reversed=True))
            #data.append(line[:-1])
            label.append(len(labels))


c2v_model = chars2vec.load_model('eng_50')
word_embeddings = c2v_model.vectorize_words(data)
print(len(word_embeddings[0]))
print(word_embeddings[123])

nrows, ncols = word_embeddings.shape
word_embeddings = word_embeddings.reshape(nrows, ncols, 1)

n_label = np.array(label)


trainX, testX, trainY, testY = train_test_split(word_embeddings, n_label, test_size = 0.2, random_state = 40, stratify=n_label)
trainY = to_categorical(trainY, num_classes=len(labels)+1)
testY = to_categorical(testY, num_classes=len(labels)+1)


l2_regul = 0.0000001


model = Sequential()

model.add(Conv1D(16, 25, padding='same', activation='relu', input_shape=(ncols, 1)))



model.add(Conv1D(32, 25, padding='same', activation='relu'))
model.add(Conv1D(64, 25, padding='same', activation='relu'))
model.add(Conv1D(128, 25, padding='same', activation='relu'))
model.add(Conv1D(512, 25, padding='same', activation='relu'))
model.add(Conv1D(1024, 25, padding='same', activation='relu'))

model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(GlobalMaxPool1D())

#model.add(Flatten())
model.add(Dense(units=512, activation='relu'))

model.add(Dropout(0.5))
model.add(BatchNormalization())




model.add(Dense(len(labels) + 1))
model.add(Activation("softmax"))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 55
checpoint = ModelCheckpoint('models/{0}_names.h5f'.format(model_name), monitor='val_loss', save_best_only=True, verbose=1)
callbacks = [checpoint]

H = model.fit(trainX, trainY, epochs=epochs, validation_data=(testX, testY), callbacks=callbacks, verbose=1)

model = load_model('models/{0}_names.h5f'.format(model_name))
score, acc = model.evaluate(testX, testY)
print(score, acc)


prediction = model.predict(testX)
prediction = prediction.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), prediction))


