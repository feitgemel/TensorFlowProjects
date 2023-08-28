import numpy as np

# load the train data :

allImagesNP = np.load("e:/temp/Unet-Animals-train-images.npy")
maskImagesNP = np.load("e:/temp/Unet-Animals-train-mask.npy")

print(allImagesNP.shape)
print(maskImagesNP.shape)

Weight = 128
Width = 128
numOfCategories = 3

# change the values of the mask from integer to categorical
from keras.utils import np_utils

# update to  categorical only for the first(!!!!) mask
# lets display the result before and after

test = maskImagesNP[0]
test = test -1 # convert the values from 1-3 to 0-2
test2 = np_utils.to_categorical(test, num_classes=numOfCategories) # 3

print(test)
print(test2)

# run the process for the all mask Numpy array
maskImagesNP = maskImagesNP - 1
maskForTheModel = np_utils.to_categorical(maskImagesNP , num_classes=numOfCategories)

print("print the type after the convert :")
print(maskForTheModel.dtype)
maskForTheModel = maskForTheModel.astype(int) # convert from float to integer
print(maskForTheModel.dtype)

# split train and test
from sklearn.model_selection import train_test_split

X_train, X_val , y_train , y_val = train_test_split(allImagesNP, maskForTheModel, test_size=0.1 , random_state=42)
print("X_train , X_val , y_train , y_val --------->>>>  shapes :")

print(X_train.shape)
print(y_train.shape)

print(X_val.shape)
print(y_val.shape)


# build the model

import tensorflow as tf
from Step02UnetModel import build_unet
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

shape = (128,128,3)
num_classes = 3
lr = 1e-4 # 0.0001
batch_size = 4
epochs = 10


model = build_unet(shape , num_classes)
print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer = tf.keras.optimizers.Adam(lr), metrics=['accuracy'])

stepsPerEpoch = np.ceil(len(X_train)/batch_size)
validationSteps = np.ceil(len(X_val)/batch_size)

best_model_file="e:/temp/Animals-Unet.h5"

callbacks = [
    ModelCheckpoint(best_model_file, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor='val_loss',patience=5 , verbose=1)
]

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data = (X_val, y_val),
                    validation_steps = validationSteps,
                    steps_per_epoch = stepsPerEpoch,
                    shuffle=True,
                    callbacks=callbacks)


# show the results 
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))


# train and validation Accuracy chart
plt.plot(epochs, acc , 'r', label="Train Accuracy")
plt.plot(epochs, val_acc, 'b' , label="Validation Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Train and Validation Accuracy")
plt.legend(loc='lower right')
plt.show()

# train and validation loss chart
plt.plot(epochs, loss , 'r', label="Train Loss")
plt.plot(epochs, val_loss, 'b' , label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Train and Validation Loss")
plt.legend(loc='upper right')
plt.show()

