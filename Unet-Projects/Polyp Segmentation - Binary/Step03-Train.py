import numpy as np

X_train = np.load("e:/temp/Unet-Polip-images-X_train.npy")
y_train = np.load("e:/temp/Unet-Polip-images-y_train.npy")
X_val = np.load("e:/temp/Unet-Polip-images-X_val.npy")
y_val = np.load("e:/temp/Unet-Polip-images-y_val.npy")
X_test = np.load("e:/temp/Unet-Polip-images-X_test.npy")
y_test = np.load("e:/temp/Unet-Polip-images-y_test.npy")

print(X_train.shape)
print(y_train.shape)

print(X_val.shape)
print(y_val.shape)

print(X_test.shape)
print(y_test.shape)


Height = 256
Width = 256

# build the model

import tensorflow as tf
from Step02BuildTheUnetModel import build_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

shape = (256,256,3)
lr = 1e-4 # 0.0001
batch_size = 8
epochs = 50


model = build_model(shape)
#print(model.summary())

opt = tf.keras.optimizers.Adam(lr)
model.compile(loss="binary_crossentropy", optimizer=opt , metrics=['accuracy'])

stepsPerEpoch = np.ceil(len(X_train) / batch_size)
validationSteps = np.ceil(len(X_val) / batch_size)

best_model_file = "e:/temp/PolypSegment.h5"

callbacks = [
            ModelCheckpoint(best_model_file, verbose=1 , save_best_only=True),
            ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1 , verbose=1, min_lr=1e-6),
            EarlyStopping(monitor="val_loss", patience=5, verbose=1) ]

history = model.fit (X_train, y_train, 
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data = (X_val, y_val),
                    validation_steps = validationSteps,
                    steps_per_epoch = stepsPerEpoch,
                    shuffle = True,
                    callbacks = callbacks )


# display the results

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']



epochs = range(len(acc))


# train and validation chart

plt.plot(epochs, acc, 'r' , label="Train accuracy")
plt.plot(epochs, val_acc, 'b' , label="Validation accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Train and validation accuracy")
plt.legend(loc='lower right')
plt.show()

# loss and validation loss chart

plt.plot(epochs, loss, 'r' , label="Train loss")
plt.plot(epochs, val_loss, 'b' , label="Validation loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Train and validation loss")
plt.legend(loc='upper right')
plt.show()


# Evaluate the model
resultEval = model.evaluate(X_test, y_test)
print("Evaluate the test data :")
print(resultEval)

