# dataset : https://www.kaggle.com/datasets/paultimothymooney/kermany2018

# Multi classs Retinal Oct classification

import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

trainPath = "E:/Data-sets/Retinal OCT Images/train"
testPath = "E:/Data-sets/Retinal OCT Images/test"
batchSize = 32

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(224,224,1))) #1 -> gray scale
model.add(MaxPooling2D())
model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(4, activation='softmax'))

print(model.summary() )

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics = ["accuracy"])


# prepare the data

datagen = ImageDataGenerator( rescale = 1./255)

trainData = datagen.flow_from_directory(trainPath,
                                        target_size=(224,224),
                                        batch_size=batchSize,
                                        color_mode='grayscale',
                                        class_mode='categorical', 
                                        shuffle=True)

testData = datagen.flow_from_directory(testPath,
                                        target_size=(224,224),
                                        batch_size=batchSize,
                                        color_mode='grayscale',
                                        class_mode='categorical', 
                                        shuffle=False)


stepsPerEpoch = np.ceil(trainData.samples / batchSize)
validationSteps = np.ceil(testData.samples / batchSize)

# Early stopping

stopEarly = EarlyStopping(monitor='val_accuracy', patience=5)

history = model.fit(trainData,
                    steps_per_epoch= stepsPerEpoch,
                    epochs=50,
                    validation_steps=validationSteps,
                    validation_data=testData,
                    callbacks=[stopEarly] )


model.save("e:/temp/retinalOCT.h5")


import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochsForPlot = range(len(acc))

# train and validation chart

plt.plot(epochsForPlot, acc, 'r' , label='Train Accuracy')
plt.plot(epochsForPlot, val_acc, 'b' , label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='lower right')
plt.show()


# loss and validation loss chart
plt.plot(epochsForPlot, loss, 'r' , label='Train Loss')
plt.plot(epochsForPlot, val_loss, 'b' , label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')
plt.show()




