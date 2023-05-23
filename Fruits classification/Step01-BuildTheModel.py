import tensorflow as tf
import os
from keras.utils import img_to_array , load_img
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D , Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

train_path = "e:/Data-sets/fruits-360_dataset/fruits-360/Training/"
test_path = "e:/Data-sets/fruits-360_dataset/fruits-360/Test/" 

BatchSize = 64 # reduce the value if you heve less Gpu memory

img = load_img(train_path + "Quince/0_100.jpg")
#plt.imshow(img)
#plt.show()

imgA = img_to_array(img)
print(imgA.shape)


# Build the model 
# ===============

model = Sequential()
model.add(Conv2D(filters=128 , kernel_size=3 , activation="relu", input_shape=(100,100,3)  ))
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=3 , activation="relu"))
model.add(Conv2D(filters=32, kernel_size=3 , activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(5000, activation="relu"))
model.add(Dense(1000, activation="relu"))
model.add(Dense(131,activation="softmax"))

print(model.summary())

# compile the model
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

# load the data

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range=0.3,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   zoom_range=0.3)


test_datagen = ImageDataGenerator(rescale= 1./255)


train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=(100,100),
                                                    batch_size=BatchSize,
                                                    color_mode="rgb",
                                                    class_mode="categorical",
                                                    shuffle=True)

test_generator = test_datagen.flow_from_directory(test_path,
                                                    target_size=(100,100),
                                                    batch_size=BatchSize,
                                                    color_mode="rgb",
                                                    class_mode="categorical")


stepsPerEpoch = np.ceil(train_generator.samples / BatchSize)
ValidationSteps = np.ceil(test_generator.samples / BatchSize)

# Early Stopping
# ==============

stop_early = EarlyStopping(monitor="val_accuracy", patience=5)


# train the model
history = model.fit(train_generator,
                    steps_per_epoch=stepsPerEpoch,
                    epochs=50,
                    validation_data=test_generator,
                    validation_steps=ValidationSteps,
                    callbacks=[stop_early])

model.save("e:/temp/Fruits360.h5")
























