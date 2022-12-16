import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint
import numpy as np
import time
import matplotlib.pyplot as plt


# final size of the images
IMG = 224
IMG_SIZE = [IMG,IMG]

numOfClasses = 144
batchSize = 64
EPOCHS=25
PATIENCE=3

trainFolder = "C:/Data-Sets/TIME -Image Dataset-Classification/train" 
testFolder = "C:/Data-Sets/TIME -Image Dataset-Classification/test"

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG,IMG,1)), # 1 Since we will you only 1 channle = grayscale
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu' ),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu' ),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(numOfClasses, activation='softmax'),

])

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

# load the data and augment it a little bit
# load the data in gray scale !!!

train_datagen = ImageDataGenerator(rescale=1./255,
                                    width_shift_range = 0.1,
                                    height_shift_range = 0.1,
                                    zoom_range = 0.2)

training_set = train_datagen.flow_from_directory(trainFolder,
                                                 shuffle=True,
                                                 target_size = IMG_SIZE,
                                                 color_mode="grayscale",
                                                 batch_size = batchSize,
                                                 class_mode = "categorical")



test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory(testFolder,
                                                 shuffle=False,
                                                 target_size = IMG_SIZE,
                                                 color_mode="grayscale",
                                                 batch_size = batchSize,
                                                 class_mode = "categorical")



stepsPerEpochs = np.ceil(training_set.samples / batchSize) # round the number
validationSteps = np.ceil(test_set.samples / batchSize)

# starting time for train
t0 = time.time()

best_model_file = "C:/Data-Sets/TIME -Image Dataset-Classification/myTimeCnn.h5"
best_model = ModelCheckpoint(best_model_file, monitor="val_accuracy", verbose=1 , save_best_only=True) 


# train the model

history = model.fit(
  training_set,
  validation_data=test_set,
  epochs=EPOCHS, # max epochs
  steps_per_epoch=stepsPerEpochs,
  validation_steps=validationSteps,
  verbose=1,
  callbacks=[best_model] )

t1 = int(time.time() - t0)

print("Total train time : " + str(t1))


# evaluate the mode (show the results)

valResult = model.evaluate(test_set)
print("Evaluate results : ")

print(valResult)
print(model.metrics_names)

#plot the results 

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss =  history.history['loss']
val_loss =  history.history['val_loss']

epochs = range(len(acc)) # the highest train epoch number

# Accuracy
plt.plot(epochs, acc , 'r', label="Training accuracy")
plt.plot(epochs, val_acc , 'b', label="Validation accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and validation Accuracy')
plt.legend(loc = 'lower right')
plt.show()

# Loss
plt.plot(epochs, loss , 'r', label="Training loss")
plt.plot(epochs, val_loss , 'b', label="Validation loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and validation Loss')
plt.legend(loc = 'upper right')
plt.show()

