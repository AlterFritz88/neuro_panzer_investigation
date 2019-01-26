from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.pipeline import Pipeline
from transliterate import translit
import pickle
from keras.utils import to_categorical
from sklearn.feature_extraction.text import  TfidfVectorizer
import os
from sklearn.metrics import classification_report
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout, GlobalMaxPooling1D, Conv1D, Embedding, GlobalMaxPool1D
from keras.layers.convolutional import MaxPooling2D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_score, make_scorer


labels = []
data_dict = {}
data = []
label = []
vectorizer = CountVectorizer()
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


#data_vectorised = vectorizer.fit_transform(data)

t = Tokenizer()
t.fit_on_texts(data)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(data)
print(encoded_docs)
n_label = np.array(label)
max_length = 10
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


trainX, testX, trainY, testY = train_test_split(padded_docs, n_label, test_size = 0.2, random_state = 42, stratify=n_label)

trainY = to_categorical(trainY, num_classes=len(labels)+1)
testY = to_categorical(testY, num_classes=len(labels)+1)


s = list(range(1, 800, 50))
units = list(range(1, 1500, 50))
param_grid = dict(a=s,
                      units=units,
                        l2_regul = [0.1, 0.05, 0.01, 0.05, 0.001, 0.005, 0.0001]
                  )

print(param_grid)
def create_model(a, units, l2_regul):
    model = Sequential()
    model.add(Embedding(vocab_size, output_dim= a, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(units=units, activation='relu', kernel_initializer="he_normal", kernel_regularizer=l2(l2_regul)))
    model.add(Dropout(0.5))
    model.add(Dense(len(labels) + 1))
    model.add(Activation("softmax"))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
epochs = 70
model = KerasClassifier(build_fn=create_model,
                            epochs=epochs, batch_size=64,
                            verbose=1)


def my_check(y_true, y_pred):
    print(precision_score( y_true.argmax(axis=1), y_pred, average="weighted"))
    return precision_score(y_pred, y_true.argmax(axis=1), average="weighted")


score = make_scorer(my_check, greater_is_better=True)



grid = GridSearchCV(model, param_grid, scoring=score, cv=10)
grid_result = grid.fit(trainX, trainY)

test_accuracy = grid.score(testX, testY)

print(grid_result.best_score_,
            grid_result.best_params_,
            test_accuracy)

