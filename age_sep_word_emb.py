from transliterate import translit
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout, GlobalMaxPooling1D, Conv1D, Embedding,BatchNormalization, GlobalMaxPool1D, CuDNNLSTM, Bidirectional, LSTM, MaxPooling1D, SeparableConv1D
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.regularizers import l2, l1, l1_l2

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt


labels = []
data_dict = {}
data = []
label = []

import re

def has_cyrillic(text):
    return bool(re.search('[а-яА-Я]', text))


with open('spisok', 'r') as file:
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
            label.append(0)
            continue

        data_dict[line[:-1]] = len(labels)
        data.append(translit(u"{}".format(line[:-1]), "ru", reversed=True))
        label.append(0)

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

        if len(labels) != 1 and len(labels) != 8 and len(labels) != 49 and len(labels) != 10:
        #if len(labels) != 2 and len(labels) != 7:
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
            label.append(1)
            continue



        data_dict[translit(u"{}".format(line[:-1]), "ru", reversed=True)] = len(labels)
        data.append(translit(u"{}".format(line[:-1]), "ru", reversed=True))
        label.append(1)
labels = ['WWII', 'Modern']


'''
for i in range(len(label)):
    print(data[i], label[i])
'''
print(label)

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
n_label = np.array(label)

trainX, testX, trainY, testY = train_test_split(data, n_label, test_size = 0.2, random_state = 40, stratify=n_label)
test_Y_nc = testY
trainY = to_categorical(trainY, num_classes=len(labels))
testY = to_categorical(testY, num_classes=len(labels))

testX_nc = testX


coded = []
for word in trainX:
    encoded_seq = [mapping[char] for char in word]
    coded.append(encoded_seq)

vocab_size = len(mapping)


trainX = pad_sequences(coded, maxlen=max_len, padding='post')


coded = []
for word in testX:
    encoded_seq = [mapping[char] for char in word]
    coded.append(encoded_seq)
vocab_size = len(mapping)


testX = pad_sequences(coded, maxlen=max_len, padding='post')
print(trainX.shape)



model = RandomForestClassifier(criterion= 'gini', max_depth= 138, max_features= 'auto', n_estimators= 1)
model.fit(trainX, trainY)
y_pred = model.predict(testX)
print('accuracy RR %s' % metrics.accuracy_score(y_pred, testY))


# neural network

l2_regul = 0.000000001
model = Sequential()
model.add(Embedding(vocab_size, output_dim= 1000, input_length=max_len, trainable=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(SeparableConv1D(512, 5, activation='relu', kernel_regularizer=l1(l2_regul)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(GlobalMaxPool1D())
model.add(Dropout(0.2))
model.add(BatchNormalization())
#model.add(Bidirectional(CuDNNLSTM(128, return_sequences=False)))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(units=256, activation='relu', kernel_regularizer=l1(l2_regul)))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(2))
model.add(Activation("softmax"))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 50
checpoint = ModelCheckpoint('models/age_names.h5f', monitor='val_loss', save_best_only=True, verbose=1)
callbacks = [checpoint]

H = model.fit(trainX, trainY, epochs=epochs, validation_data=(testX, testY), callbacks=callbacks, verbose=1)

model = load_model('models/age_names.h5f')
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


y_probs = model.predict(testX)

# Get predicted labels for test dataset
y_preds = y_probs.argmax(axis=1)


# Indices corresponding to test images which were mislabeled


bad_test_idxs = np.where(testY!=y_preds)

for i in range(len(testY)):
    if test_Y_nc[i] != y_preds[i]:
        print(testX_nc[i])
        print(test_Y_nc[i], y_preds[i])