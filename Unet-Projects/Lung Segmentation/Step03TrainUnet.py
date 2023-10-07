import numpy as np

# load the data 
print("start lodaing ")
allImagesNp = np.load("e:/temp/Unet-Train-Lung-Images.npy")
maskImagesNp = np.load("e:/temp/Unet-Train-Lung-Masks.npy")

allValidateImagesNP = np.load("e:/temp/Unet-Validate-Lung-Images.npy") 
maskValidateImagesNP = np.load("e:/temp/Unet-Validate-Lung-Masks.npy") 

print(allImagesNp.shape)
print(maskImagesNp.shape)
print(allValidateImagesNP.shape)
print(maskValidateImagesNP.shape)

Height = 256
Width = 256

# build the model
import tensorflow as tf
from Step02Model import build_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

shape=(256, 256, 3)
lr = 1e-4 # 0.0001
batchSize = 4
epochs = 50

model = build_model(shape)
print(model.summary())


opt = tf.keras.optimizers.Adam(lr)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

stepsPerEpoch = np.ceil( len(allImagesNp) / batchSize)
validationSteps = np.ceil( len(allValidateImagesNP) / batchSize)

best_model_file = "e:/temp/lung-Unet.h5"

callbacks = [
    ModelCheckpoint(best_model_file, verbose=1 , save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1 , verbose=1, min_lr=1e-7),
    EarlyStopping(monitor="val_accuracy", patience=20 ,verbose=1 )] 


history = model.fit(allImagesNp, maskImagesNp,
                    batch_size = batchSize,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(allValidateImagesNP, maskValidateImagesNP),
                    validation_steps= validationSteps,
                    steps_per_epoch = stepsPerEpoch,
                    shuffle=True,
                    callbacks=callbacks  )


