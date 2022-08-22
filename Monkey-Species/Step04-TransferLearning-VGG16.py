from keras.layers import Dense , Flatten #-> for the last layers
from keras.models import Model

#VGG16 model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint

# image augmentaion
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

import numpy as np 
from glob import glob 
import matplotlib.pyplot as plt 
import time

# Parameters
IMG=200
IMG_SIZE = [IMG, IMG]

numOfClasses = 10
batchSize = 32
EPOCHS = 30
PATIENCE=5

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

# The transfer learning - tune the VGG16 model
myVgg = VGG16(input_shape=IMG_SIZE+ [3],
            weights='imagenet',
            include_top=False) # False means , remove the last fully coneccted layers

#print(myVgg.summary())

# we freeze the layers -> we dont need training
for layer in myVgg.layers:
    layer.trainable = False

# add Flatten layer

PlusFlattenLayer = Flatten()(myVgg.output)

# add the last layer
lastPredictionLayer = Dense(numOfClasses, activation='softmax')(PlusFlattenLayer)


# final model 
model = Model(inputs=myVgg.input , outputs=lastPredictionLayer)
print(model.summary())

model.compile(loss='categorical_crossentropy',
            optimizer='Adam',
            metrics=['accuracy'] )

best_model_file = "C:/Python-cannot-upload-to-GitHub/10-Monkey-Species/myTransferLearningMonkeyModel.h5"
bestModel = ModelCheckpoint(best_model_file, monitor='val_accuracy', verbose=1, save_best_only=True)

# train the model
history = model.fit( training_set,
                    validation_data = test_set,
                    epochs=EPOCHS,
                    steps_per_epoch=stepsPerEpochs,
                    validation_steps=validationSteps,
                    verbose=1,
                    callbacks=[bestModel])


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

