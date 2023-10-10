import numpy as np
import cv2

# load the saved data :
allImages = np.load("e:/temp/Malaria-images.npy")
allLables = np.load("e:/temp/Malaria-lables.npy")


print(allImages.shape)
print(allLables.shape)

input_shape=(124,124,3)
shape=(124,124)

# show the first image
img = allImages[0]
label = allLables[0]
print(label)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# prepare all the data
# normalize values between 0 and 1

allImagesForModel = allImages / 255.0

# split train and test
from sklearn.model_selection import train_test_split

print("Split train and test data : ")
X_train, X_test, y_train, y_test = train_test_split(allImagesForModel, allLables, test_size=0.3 , random_state=42)

print("X_train , X_test , y_train , y_test ----->>> shapes :")

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Conv2D(input_shape=input_shape, filters=16 , kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=16, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())

model.add(Dense(1024, activation="relu"))
model.add(Dense(1,activation="sigmoid"))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

batch = 32
epochs = 10

stepsPerEpoch = np.ceil(len(X_train)/ batch)
validationSteps = np.ceil(len(X_test)/ batch)

best_model_file = "e:/temp/Malaria_binary.h5"

best_model = ModelCheckpoint(best_model_file,monitor="val_accuracy", verbose=1, save_best_only=True)

history = model.fit(X_train, y_train,
                    batch_size=batch,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    validation_steps = validationSteps,
                    steps_per_epoch = stepsPerEpoch,
                    shuffle=True,
                    callbacks=[best_model]   )










