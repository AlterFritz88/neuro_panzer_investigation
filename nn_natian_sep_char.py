from transliterate import translit
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout, GlobalMaxPooling1D, Conv1D, Embedding, GlobalMaxPool1D, CuDNNLSTM, Bidirectional, LSTM, MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np

from sklearn.ensemble import RandomForestClassifier

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
        print(len(labels))

        if train == 1:
            if len(labels) != 1 and len(labels) != 8 and len(labels) != 49 and len(labels) != 10:
                sent = line[:-1].split(' ')
                print(line[:-1])
                sent_new = []
                for t in sent:
                    if has_cyrillic(t) == False:
                        sent_new.append(t)
                sent_st = ' '.join(sent_new)
                if len(sent_st) < 2:
                    continue
                print(sent_st)
                data_dict[sent_st] = len(labels)
                data.append(sent_st)
                label.append(len(labels))
                continue
            data_dict[line[:-1]] = len(labels)
            data.append(translit(u"{}".format(line[:-1]), "ru", reversed=True))
            # data.append(line[:-1])
            label.append(len(labels))
        #if len(labels) != 1 and len(labels) != 8 and len(labels) != 49 and len(labels) != 10:
        if train == 0:
            if len(labels) != 2 and len(labels) != 7:
                sent = line[:-1].split(' ')
                print(line[:-1])
                sent_new = []
                for t in sent:
                    if has_cyrillic(t) == False:
                        sent_new.append(t)
                sent_st = ' '.join(sent_new)
                if len(sent_st) < 2:
                    continue
                print(sent_st)
                data_dict[sent_st] = len(labels)
                data.append(sent_st)
                label.append(len(labels))
                continue


            data_dict[line[:-1]] = len(labels)
            data.append(translit(u"{}".format(line[:-1]), "ru", reversed=True))
            #data.append(line[:-1])
            label.append(len(labels))


print(data)
data_string = ''
max_len = 0
for i in data:
    data_string += i
    if len(i) > max_len:
        max_len = len(i)
print(data_string)
print(max_len)


chars = sorted(list(set(data_string)))
mapping = dict((c, i) for i, c in enumerate(chars))


coded = []
for word in data:
    encoded_seq = [mapping[char] for char in word]
    coded.append(encoded_seq)

vocab_size = len(mapping)
print(vocab_size)

padded = pad_sequences(coded, maxlen=max_len, padding='post')
print(padded[1])
#padded = padded * 10
print(padded[1])
n_label = np.array(label)

trainX, testX, trainY, testY = train_test_split(padded, n_label, test_size = 0.2, random_state = 42, stratify=n_label)

trainY = to_categorical(trainY, num_classes=len(labels)+1)
testY = to_categorical(testY, num_classes=len(labels)+1)



model = RandomForestClassifier(criterion= 'gini', max_depth= 138, max_features= 'auto', n_estimators= 1)
model.fit(trainX, trainY)
y_pred = model.predict(testX)
print('accuracy RR %s' % metrics.accuracy_score(y_pred, testY))


# neural network


model = Sequential()
model.add(Embedding(vocab_size, output_dim= 1000, input_length=max_len, trainable=True))

model.add(Conv1D(512, 5, activation='relu'))
model.add(GlobalMaxPool1D())

#model.add(Bidirectional(CuDNNLSTM(128, return_sequences=False)))
model.add(Dropout(0.3))
#model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))



model.add(Dense(len(labels) + 1))
model.add(Activation("softmax"))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 15
checpoint = ModelCheckpoint('models/{0}_names.h5f'.format(model_name), monitor='val_loss', save_best_only=True, verbose=1)
callbacks = [checpoint]

H = model.fit(trainX, trainY, epochs=epochs, validation_data=(testX, testY), callbacks=callbacks, verbose=1)

model = load_model('models/{0}_names.h5f'.format(model_name))
score, acc = model.evaluate(testX, testY)
print(score, acc)


prediction = model.predict(testX)
prediction = prediction.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), prediction))


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('graph_text')