# Dataset : https://www.kaggle.com/datasets/cmglonly/simple-dinosurus-dataset

# 5 Classes of Dinosaurs

import numpy as np
import cv2
import os

path = "E:/Data-sets/Simple Dinosaur Dataset"
categories = os.listdir(path)
categories.sort()
print(categories)


numOfClasses = len(categories)
print("Number of categories :")
print(numOfClasses)

batchSize = 32
imageSize = (224,224)


# prepare the data 
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale= 1./255 , validation_split=0.2 , horizontal_flip=True)

train_dataset = datagen.flow_from_directory(batch_size=batchSize,
                                            directory=path,
                                            color_mode='rgb',
                                            shuffle=True,
                                            target_size=imageSize,
                                            subset="training",
                                            class_mode="categorical")


validation_dataset = datagen.flow_from_directory(batch_size=batchSize,
                                            directory=path,
                                            color_mode='rgb',
                                            shuffle=True,
                                            target_size=imageSize,
                                            subset="validation",
                                            class_mode="categorical")


batch_x, batch_y = next(train_dataset)

# print the shapes of the first batch ( images and labels)
print('Batch of images shape ', batch_x.shape)
print('Batch of images label ', batch_y.shape)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense



model = Sequential()
model.add(Conv2D(filters=16 , kernel_size=3 , activation='relu', padding='same', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))

model.add(Conv2D(filters=32 , kernel_size=3 , activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64 , kernel_size=3 , activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(numOfClasses, activation='softmax'))


print(model.summary())


model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

stepsPerEpochs = np.ceil(train_dataset.samples / batchSize)
validationSteps =np.ceil(validation_dataset.samples / batchSize)

# Early stop
from keras.callbacks import ModelCheckpoint

best_model_file = "e:/temp/dino.h5"
best_model = ModelCheckpoint(best_model_file , monitor='val_accuracy', verbose=1, save_best_only=True)

history = model.fit(train_dataset,
                steps_per_epoch=stepsPerEpochs,
                epochs=50,
                validation_data=validation_dataset,
                validation_steps=validationSteps,
                callbacks=[best_model] )

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochsForGraph = range(len(acc))

# plot the train and validation

plt.plot(epochsForGraph, acc, 'r' , label="Train accuracy")
plt.plot(epochsForGraph, val_acc, 'b' , label="Validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Train and Validation Accuracy")
plt.legend(loc='lower right')
plt.show()


# plot loss and validation loss 

plt.plot(epochsForGraph, loss, 'r' , label="Train loss")
plt.plot(epochsForGraph, val_loss, 'b' , label="Validation loss")
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title("Train and Validation Loss")
plt.legend(loc='upper right')
plt.show()









