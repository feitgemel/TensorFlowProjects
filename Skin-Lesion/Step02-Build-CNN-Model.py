import numpy as np
import pandas as pd
import cv2

#load the data

allImages = np.load("C:/Data-Sets/Skin-Lesion/allImages64.npy")
allLables = np.load("C:/Data-Sets/Skin-Lesion/allLables.npy")

print(allImages.shape)
print(allLables.shape)

# categories 
categories = ['MEL', 'NV', 'BCC']

input_shape = (64,64,3)
numofCategories = len(categories)
# show the first image

#img = allImages[1]
#label = allLables[1]
#imgCategory = categories[label]

#print(img.shape)
#print(imgCategory)
#cv2.putText(img, imgCategory, (0,50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0,), 2)
#cv2.imshow("img",img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# convert the lables to categorical
from keras.utils import np_utils
allLablesForModel = np_utils.to_categorical(allLables, num_classes=numofCategories)
print(allLablesForModel)

# normalize the images between 0 to 1
allImagesForModel = allImages / 255.0

# split train and test
from sklearn.model_selection import train_test_split

X_train, X_test , y_train, y_test = train_test_split(allImagesForModel, allLablesForModel, test_size=0.3, random_state=42)

print("Results - shapes : ")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# build the model

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint


model = Sequential()

model.add(Conv2D(input_shape=input_shape, filters=9 , kernel_size=(3,3), padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dense(numofCategories, activation='softmax'))

print(model.summary())

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

batch=32
epochs= 20

stepsPerEpoch = np.ceil(len(X_train) / batch)
validationSteps = np.ceil(len(X_test) / batch)

# save only the best model

best_model_file = "C:/Data-Sets/Skin-Lesion/best.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_accuracy', verbose=1, save_best_only=True)


# train the model

history = model.fit(X_train, y_train,
                    batch_size=batch,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test,y_test),
                    validation_steps=validationSteps,
                    steps_per_epoch=stepsPerEpoch,
                    shuffle=True,
                    callbacks=[best_model])


resultEval = model.evaluate(X_test, y_test)
print(resultEval)





