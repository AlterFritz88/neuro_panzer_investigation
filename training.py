from imutils import paths
from keras.preprocessing.image import img_to_array, array_to_img
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import classification_report, precision_score
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.regularizers import l2, l1, l1_l2



class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
    def preprocess(self, image):
        image = cv2.resize(image, (self.width, self.height), interpolation=self.inter)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

dataset_path = 'dataset'

data = []
labels = []
preproc = SimplePreprocessor(92,92)
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(dataset_path)))
images_naturals = []
list_labels = os.listdir(path=dataset_path)
print(list_labels)
# loop over the input images
label_dict ={}
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    print(imagePath)
    image = preproc.preprocess(image)
    image_natural = image
    image = img_to_array(image)
    data.append(image)
    images_naturals.append(image_natural)

    # extract the class label from the image path and update the
	# labels list
    number_labels = len(list_labels)
    label_dir = imagePath.split(os.path.sep)[-2]
    print(label_dir)
    label = list_labels.index(label_dir)
    label_dict[label_dir] = label
    print(label)
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
#data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
classTotals = to_categorical(labels, num_classes=number_labels).sum(axis=0)
classWeight = classTotals.max() / classTotals

print(labels)
with open('models/{0}_dict_labels'.format(dataset_path), 'w') as file:
    for key, value in label_dict.items():
        file.write(key + ' ' + str(value) + '\n')

trainX, testX, trainY, testY = train_test_split(data, labels, test_size = 0.25, random_state = 42, stratify=labels)

# convert the labels from integers to vectors
test_Y_nc = testY  #для сохраненния на диск проб
testX_nc = testX

trainY = to_categorical(trainY, num_classes=number_labels)
testY = to_categorical(testY, num_classes=number_labels)

trainX = np.array(trainX, dtype="float") / 255.0
testX = np.array(testX, dtype="float") / 255.0

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout, BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam

l2s = [0.005, 0.01, 0.001, 0.0005,  0.00005, 0.00001]
answers = []

epochs = 30


def model1():
    lr = 0.01
    l2_regul = 0.001
    l1_r = 1e-06
    model = Sequential()

    model.add(Conv2D(16, (7, 7), padding="same",
                kernel_initializer="he_normal", kernel_regularizer=l2(l2_regul),
                input_shape=(92, 92, 1)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(l2_regul)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(l2_regul)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(l2_regul)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(l2_regul)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(l2_regul)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(l2_regul)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(l2_regul)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(l2_regul)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



    model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(l2_regul)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(l2_regul)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(l2_regul)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(Dropout(0.25))






    model.add(Flatten())
    model.add(Dense(512, kernel_initializer="he_normal", kernel_regularizer=l2(l2_regul)))
    model.add(Activation("relu"))
    #model.add(Dropout(0.3))

    model.add(BatchNormalization())
    #model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(number_labels))
    model.add(Activation("softmax"))

    opt = SGD(lr=lr, decay=lr / epochs, momentum=0.9, nesterov=True)  # decay=0.003/epochs
    ad = Adam(lr=lr, decay=lr / epochs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # loss=binary_crossentropy''
    return model

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.constraints import maxnorm
def model2():
    lr = 0.05
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',
                            W_constraint=maxnorm(3)))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',
                            W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',
                            W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',
                            W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))
    model.add(Dense(number_labels))
    model.add(Activation("softmax"))
    opt = SGD(lr=lr, decay=lr / epochs, momentum=0.9, nesterov=True)  # decay=0.003/epochs
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # loss=binary_crossentropy''
    return model

checpoint = ModelCheckpoint('models/{0}_test.h5f'.format(dataset_path), monitor='val_loss', save_best_only=True, verbose=1)
callbacks = [checpoint]


aug = ImageDataGenerator(rotation_range=7, width_shift_range=[-0.2, 0, +0.2],
    height_shift_range=[-0.1, 0, +0.1], shear_range=0.2, zoom_range=0.3,
    horizontal_flip=True, fill_mode="constant")

model = model1()

#H = model.fit_generator(aug.flow(trainX, trainY, batch_size=128), validation_data=(testX, testY), steps_per_epoch=len(trainX) // 3, epochs=epochs, verbose=1, callbacks=callbacks, class_weight=classWeight)

import time
for i in range(7):
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=128), validation_data=(testX, testY),
                            steps_per_epoch=len(trainX) // 3, epochs=5, verbose=1, callbacks=callbacks,
                            class_weight=classWeight)
    model = load_model('models/{0}_test.h5f'.format(dataset_path))
    print(i, 'sleeping')
    time.sleep(350)

#H = model.fit(trainX, trainY, epochs=epochs, validation_data=(testX, testY), verbose=1, batch_size=128, callbacks=callbacks, class_weight=classWeight)
model = load_model('models/{0}_test.h5f'.format(dataset_path))
#score, acc = model.evaluate(testX, testY, batch_size=164)
#print(score, acc)

prediction = model.predict(testX)
prediction = prediction.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), prediction, target_names=list_labels))
print(precision_score(testY.argmax(axis=1), prediction, average="weighted"))
#answers.append((l2_regul, precision_score(testY.argmax(axis=1), prediction, average="weighted")))
print(answers)
#model.save('model_tanks')
print(answers)
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
plt.savefig('graph')

y_probs = model.predict(testX)

# Get predicted labels for test dataset
y_preds = y_probs.argmax(axis=1)
print(len(y_preds))
print(len(test_Y_nc))

# Indices corresponding to test images which were mislabeled

'''
bad_test_idxs = np.where(testY!=y_preds)
path_bad_img = 'bad_predict'
for i in range(len(test_Y_nc)):
    if test_Y_nc[i] != y_preds[i]:
        print(test_Y_nc[i], y_preds[i], 'не правда')
        img_s = array_to_img(testX_nc[i])
        #imsave('{0}/{1}'.format(path_bad_img, i),  array_to_img(testX_nc[i]))
        img_s.save('{0}/{1}---{2}--{3}.png'.format(path_bad_img, list_labels[test_Y_nc[i]], list_labels[y_preds[i]], i ))

'''




