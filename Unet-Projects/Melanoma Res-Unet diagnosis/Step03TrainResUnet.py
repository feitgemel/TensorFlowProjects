import numpy as np

#load the data

print("start loading the train data :")
allImagesNP = np.load("e:/temp/Unet-Train-Melanoa-Images.npy")
maskImagesNP  = np.load("e:/temp/Unet-Train-Melanoa-Masks.npy")

print("start loading the validation data :")


allValidateImageNP = np.load("e:/temp/Unet-Test-Melanoa-Images.npy")
maskValidateImages = np.load("e:/temp/Unet-Test-Melanoa-Masks.npy")

print("Finish save the Data ..........................")

print(allImagesNP.shape)
print(maskImagesNP.shape)
print(allValidateImageNP.shape)
print(maskValidateImages.shape)

Height = 128
Width = 128

# build the model :

import tensorflow as tf
from Step02BuildResUnetModel import build_resunet
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

shape = (128, 128, 3)
lr = 1e-4 # 0.0001
batch_size = 8
epochs = 50

model = build_resunet(shape)
print(model.summary())

opt = tf.keras.optimizers.Adam(lr)
model.compile(loss="binary_crossentropy", optimizer = opt , metrics=['accuracy'])

stepsPerEpoch = np.ceil(len(allImagesNP)/batch_size)
validationSteps = np.ceil(len(allValidateImageNP)/batch_size)

best_model_file = "e:/temp/MelanomaResUnet.h5"

callbacks = [
    ModelCheckpoint(best_model_file,verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1, verbose=1, min_lr=1e-7),
    EarlyStopping(monitor="val_accuracy", patience=20, verbose=1) ]


history = model.fit(allImagesNP, maskImagesNP,
                    batch_size = batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(allValidateImageNP, maskValidateImages),
                    validation_steps = validationSteps,
                    steps_per_epoch = stepsPerEpoch,
                    shuffle=True,
                    callbacks=callbacks )





