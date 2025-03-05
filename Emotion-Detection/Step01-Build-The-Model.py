# Dataset : https://www.kaggle.com/datasets/msambare/fer2013

import numpy as np
import tensorflow as tf
import cv2
import os

trainPath = "C:/Data-Sets/Emotion-Faces/train"
testPath  = "C:/Data-Sets/Emotion-Faces/test"

folderList = os.listdir(trainPath)
folderList.sort()

print(folderList)

X_train = []
y_train = []

X_test=[]
y_test=[]

# load the train data into arrays 

for i , category in enumerate(folderList):
    files = os.listdir(trainPath+"/"+category)
    for file in files:
        print(category+"/"+file)
        img = cv2.imread(trainPath+"/"+category+'/{0}'.format(file),0)
        X_train.append(img)
        y_train.append(i) # each folder will be a number 

print(len(X_train)) # 28709 train images

# show the first image
#img1 = X_train[0]
#cv2.imshow("img1",img1)
#cv2.waitKey(0)

# check the labels
print(y_train)
print(len(y_train))


# do the same for the test data
folderList = os.listdir(testPath)
folderList.sort()

for i , category in enumerate(folderList):
    files = os.listdir(testPath+"/"+category)
    for file in files:
        print(category+"/"+file)
        img = cv2.imread(testPath+"/"+category+'/{0}'.format(file),0)
        X_test.append(img)
        y_test.append(i) # each folder will be a number 

print("Test data :")
print(len(X_test)) 
print(len(y_test)) 

# convert the data to numpy

X_train = np.array(X_train, 'float32')
y_train = np.array(y_train, 'float32')
X_test = np.array(X_test, 'float32')
y_test = np.array(y_test, 'float32')

#check 
print(X_train.shape)
print(X_train[0])

# two tasks :
# normalize the image : 0 to 1
# add another dimention to the data : (28709, 48, 48 , 1) 

X_train = X_train / 255.0
X_test = X_test / 255.0

#reshape the train data 

numOfImages = X_train.shape[0] # 28709
X_train = X_train.reshape(numOfImages,48,48,1) # add another dim for gray image

print(X_train[0])
print(X_train.shape)

# the same for the test
numOfImages = X_test.shape[0]
X_test = X_test.reshape(numOfImages,48,48,1)
print(X_test.shape)

# convert the lables to categorical 
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train, num_classes=7)
y_test = np_utils.to_categorical(y_test, num_classes=7)

print("To categorical:")
print(y_train)
print(y_train.shape)
print(y_train[0])

# Build the model :
# =================

input_shape = X_train.shape[1:]
print(input_shape)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

model = Sequential()
model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu"))
model.add(Dense(7,activation="softmax"))

print(model.summary())

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

batch=32
epochs=30

stepsPerEpoch = np.ceil(len(X_train)/batch)
validationSteps = np.ceil(len(X_test)/batch)

stopEarly = EarlyStopping(monitor='val_accuracy' , patience=5)

# train the model
history = model.fit(X_train,
                    y_train,
                    batch_size=batch,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test,y_test),
                    shuffle=True,
                    callbacks=[stopEarly])


# show the result based on pyplot

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# show the charts
epochs = range(len(acc))

# show train and validation train chart

plt.plot(epochs, acc , 'r' , label="Train accuracy")
plt.plot(epochs, val_acc , 'b' , label="Validation accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Trainig and validation Accuracy")
plt.legend(loc='lower right')
plt.show()

# show loss and validation loss chart

plt.plot(epochs, loss , 'r' , label="Train loss")
plt.plot(epochs, val_loss , 'b' , label="Validation loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Trainig and validation Loss")
plt.legend(loc='upper right')
plt.show()

# save the model 
modelFileName = "e:/temp/emotion.h5"
model.save(modelFileName)



