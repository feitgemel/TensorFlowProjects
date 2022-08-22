# pip install tensorflow

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint
import numpy as np
import time
import matplotlib.pyplot as plt

# Parameters
IMG=200
IMG_SIZE = [IMG, IMG]

numOfClasses = 10
batchSize = 32
EPOCHS = 30



# build the model :

model = tf.keras.models.Sequential ([

    tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=(IMG,IMG,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(numOfClasses, activation='softmax')


])

print(model.summary())


# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Data
# =====

trainMyImagesFolder = "C:/Python-cannot-upload-to-GitHub/10-Monkey-Species/training/training"
testMyImagesFolder = "C:/Python-cannot-upload-to-GitHub/10-Monkey-Species/validation/validation"

train_datagen = ImageDataGenerator(rescale = 1. / 255, 
                                    rotation_range = 20 ,
                                    width_shift_range = 0.2 ,
                                    height_shift_range = 0.2 ,
                                    shear_range = 0.2 ,
                                    zoom_range = 0.2 ,
                                    horizontal_flip = True)

training_set = train_datagen.flow_from_directory(trainMyImagesFolder,
                                                shuffle=True,
                                                target_size=IMG_SIZE,
                                                batch_size=batchSize,
                                                class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1. / 255)


test_set = test_datagen.flow_from_directory(testMyImagesFolder,
                                                shuffle=False, #### important
                                                target_size=IMG_SIZE,
                                                batch_size=batchSize,
                                                class_mode = 'categorical')


stepsPerEpochs = np.ceil (training_set.samples / batchSize) # round the result up
validationSteps =np.ceil (test_set.samples / batchSize) 

best_model_file = "C:/Python-cannot-upload-to-GitHub/10-Monkey-Species/myCnnMonkeyModel.h5"
bestModel = ModelCheckpoint(best_model_file, monitor='val_accuracy', verbose=1, save_best_only=True)


# fit the model

history = model.fit (
    training_set,
    validation_data = test_set,
    epochs = EPOCHS,
    steps_per_epoch = stepsPerEpochs,
    validation_steps = validationSteps,
    verbose=1,
    callbacks=[bestModel])


# evaluate the model 
valResults = model.evaluate(test_set)
print(valResults)
print(model.metrics_names)

# display the results on charts

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

actualEpochs = range(len(acc))
print("Actual Epochs : "+ str(actualEpochs))

plt.plot(actualEpochs, acc , 'r', label='Training accuracy')
plt.plot(actualEpochs, val_acc , 'b', label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy')

plt.show()

