from transliterate import translit
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout, GlobalMaxPooling1D, Conv1D, Embedding, GlobalMaxPool1D, SeparableConv1D, CuDNNLSTM, Bidirectional, LSTM, BatchNormalization, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, Model
from keras.regularizers import l2, l1, l1_l2
from keras.layers.merge import concatenate

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt

import re

def has_cyrillic(text):
    return bool(re.search('[а-яА-Я]', text))

train = 1  # 1 = modern    0 = wwii

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


data_string = ''
max_len = 0
for i in data:
    data_string += i
    if len(i) > max_len:
        max_len = len(i)



chars = sorted(list(set(data_string)))
print(chars)
mapping = dict((c, i) for i, c in enumerate(chars))


coded = []
for word in data:
    encoded_seq = [mapping[char] for char in word]
    coded.append(encoded_seq)

vocab_size = len(mapping)

padded = pad_sequences(coded, maxlen=max_len, padding='post')


n_label = np.array(label)

trainX, testX, trainY, testY = train_test_split(data, n_label, test_size = 0.2, random_state = 40, stratify=n_label)
test_Y_nc = testY
trainY = to_categorical(trainY, num_classes=len(labels)+1)
testY = to_categorical(testY, num_classes=len(labels)+1)

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
print(vocab_size)

testX = pad_sequences(coded, maxlen=max_len, padding='post')




model = RandomForestClassifier(criterion= 'gini', max_depth= 138, max_features= 'auto', n_estimators= 1)
model.fit(trainX, trainY)
y_pred = model.predict(testX)
print('accuracy RR %s' % metrics.accuracy_score(y_pred, testY))

'''
gnb = GaussianNB()
gnb.fit(trainX, trainY)
y_pred = gnb.predict(testX)
print('accuracy RR %s' % metrics.accuracy_score(y_pred, testY))
'''


# neural network


l2_regul = 0.0000001

model = Sequential()
model.add(Embedding(vocab_size + 1, output_dim= 1000, input_length=max_len, trainable=True))

model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(SeparableConv1D(512, 5, padding='same', activation='relu', kernel_regularizer=l1(l2_regul)))

model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(GlobalMaxPool1D())

#model.add(Bidirectional(CuDNNLSTM(128, return_sequences=False)))
model.add(Dropout(0.3))
model.add(BatchNormalization())

#model.add(Flatten())
model.add(Dense(units=256, activation='relu', kernel_regularizer=l1(l2_regul)))

model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(len(labels) + 1))
model.add(Activation("softmax"))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def model_big():
    inputs1 = Input(shape=(max_len,))
    embed1 = Embedding(vocab_size + 1, 1000, trainable=True)(inputs1)
    conv1 = Conv1D(filters=500, kernel_size=2, activation='relu')(embed1)
    drop1 = Dropout(0.1)(conv1)
    norm1 = BatchNormalization()(drop1)
    pool1 = GlobalMaxPool1D()(norm1)
    #flat1 = Flatten(pool1)

    inputs2 = Input(shape=(max_len,))
    embed2 = Embedding(vocab_size + 1, 1000,trainable=True)(inputs2)
    conv2 = Conv1D(filters=500, kernel_size=5, activation='relu')(embed2)
    drop2 = Dropout(0.1)(conv2)
    norm2 = BatchNormalization()(drop2)
    pool2 = GlobalMaxPool1D()(norm2)
    #flat2 = Flatten(pool2)

    inputs3 = Input(shape=(max_len,))
    embed3 = Embedding(vocab_size + 1, 1000,trainable=True)(inputs3)
    conv3 = Conv1D(filters=500, kernel_size=7, activation='relu')(embed3)
    drop3 = Dropout(0.1)(conv3)
    norm3 = BatchNormalization()(drop3)
    pool3 = GlobalMaxPool1D()(norm3)
    #flat3 = Flatten(pool3)

    merged = concatenate([pool1, pool2, pool3])
    dense1 = Dense(units=512, activation='relu')(merged)
    drop_glob = Dropout(0.3)(dense1)
    norm_glob = BatchNormalization()(drop_glob)
    output = Dense(len(labels) + 1, activation='softmax')(norm_glob)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

epochs = 55
checpoint = ModelCheckpoint('models/{0}_names.h5f'.format(model_name), monitor='val_loss', save_best_only=True, verbose=1)
callbacks = [checpoint]

#model = model_big()

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


y_probs = model.predict(testX)

# Get predicted labels for test dataset
y_preds = y_probs.argmax(axis=1)


# Indices corresponding to test images which were mislabeled


bad_test_idxs = np.where(testY!=y_preds)

for i in range(len(testY)):
    if test_Y_nc[i] != y_preds[i]:
        print(testX_nc[i])
        print(test_Y_nc[i], y_preds[i])