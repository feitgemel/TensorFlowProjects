import numpy as np

# load the saved Numpy arrayes (train and test data)

print("Load the Train and Test Data :")
allImagesNP = np.load("e:/temp/Unet-Train-Melanoa-Images.npy")
maskImagesNP = np.load("e:/temp/Unet-Train-Melanoa-Masks.npy")
allTestImagesNP = np.load("e:/temp/Unet-Test-Melanoa-Images.npy")
maskTestImagesNP = np.load("e:/temp/Unet-Test-Melanoa-Masks.npy")


print(allImagesNP.shape)
print(maskImagesNP.shape)
print(allTestImagesNP.shape)
print(maskTestImagesNP.shape)

Height=128
Width=128

# build the model

import tensorflow as tf
from Step02Model import build_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

shape = (128,128,3)

lr = 1e-4 # 0.001
batch_size = 8
epochs = 50

model = build_model(shape)
print(model.summary())

opt = tf.keras.optimizers.Adam(lr)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

stepsPerEpoch = np.ceil(len(allImagesNP) / batch_size) # round up the result
validationSteps = np.ceil(len(allTestImagesNP) / batch_size) # round up the result

best_model_file = "e:/temp/melanoma-Unet.h5"

callbacks = [
        ModelCheckpoint(best_model_file, verbose=1 , save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.1, verbose=1, min_lr=1e-7),
        EarlyStopping(monitor="val_accuracy", patience=20 , verbose=1) ]

history = model.fit(allImagesNP, maskImagesNP,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(allTestImagesNP, maskTestImagesNP),
                    steps_per_epoch=stepsPerEpoch,
                    validation_steps=validationSteps,shuffle=True,
                    callbacks=callbacks)



