import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

import tensorflow
from tensorflow import keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

shape=(1025,97)
batchSize = 32

imageGenerator = ImageDataGenerator(rescale=1. / 255 , validation_split=0.2)

train_dataset = imageGenerator.flow_from_directory(directory="C:/Data-Sets/Musical Chord Classification/New",
                                                            batch_size=batchSize,
                                                            target_size=shape,
                                                            subset="training",
                                                            color_mode="grayscale",
                                                            class_mode="binary")

validation_dataset = imageGenerator.flow_from_directory(directory="C:/Data-Sets/Musical Chord Classification/New",
                                                            batch_size=batchSize,
                                                            target_size=shape,
                                                            subset="validation",
                                                            color_mode="grayscale",
                                                            class_mode="binary")


#show a sample image
batch1 = train_dataset[0]

# get image number 5

img = batch1[0][5]
lab = batch1[1][5]

print(img.shape)
plt.imshow(img)
plt.title(lab)
plt.axis('off')
plt.show()

# model

optimizer = keras.optimizers.Adam(learning_rate=0.0001)

model = keras.models.Sequential([
keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=[1025, 97, 1]),
keras.layers.MaxPooling2D(pool_size=2, strides=2),

keras.layers.Conv2D(128, (3,3), activation='relu'),
keras.layers.MaxPooling2D(pool_size=2, strides=2),

keras.layers.Conv2D(256, (3,3), activation='relu'),
keras.layers.MaxPooling2D(pool_size=2, strides=2),

keras.layers.Conv2D(512, (3,3), activation='relu'),
keras.layers.MaxPooling2D(pool_size=2, strides=2),

keras.layers.Flatten(),
keras.layers.Dense(1024, activation='relu'),

keras.layers.Dense(1, activation='sigmoid') ])


model.compile(loss='BinaryCrossentropy', optimizer=optimizer, metrics=['accuracy'])

stepsPerEpochs = np.ceil(train_dataset.samples / batchSize)
validationSteps = np.ceil(validation_dataset.samples / batchSize)


best_model_file = "e:/temp/Audio-Mijor-Minor.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_accuracy', verbose=1, save_best_only=True)


# train the model

model.fit(train_dataset,
          steps_per_epoch=stepsPerEpochs,
          epochs=200,
          validation_data=validation_dataset,
          validation_steps=validationSteps,
          callbacks=[best_model])






