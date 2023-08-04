import numpy as np

IMAGE_SIZE = (331,331)
IMAGE_FULL_SIZE = (331,331,3)
batchSize = 8

allImages = np.load("e:/temp/allDogsImages.npy")
allLabes = np.load("e:/temp/allDogsLables.npy")

print(allImages.shape)
print(allLabes.shape)

# convert the lables text to integers
print(allLabes)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
integerLables = le.fit_transform(allLabes)
print(integerLables)

# unique interger lables
numOfCategories = len(np.unique(integerLables))  # = 120
print(numOfCategories)

# convert the interg lables to categorical -> prepare for the train
from tensorflow.keras.utils import to_categorical

allLablesForModel = to_categorical(integerLables, num_classes = numOfCategories)
print(allLablesForModel)

# normelize the images from 0-255 to 0-1
allImagesForModel = allImages / 255.0


# create train and test data
from sklearn.model_selection import train_test_split

print("Before split train and test :")

X_train , X_test , y_train , y_test = train_test_split(allImagesForModel, allLablesForModel, test_size=0.3, random_state=42)

print("X_train , X_test , y_train , y_test -----> shapes :")

print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)


# free some memory
del allImages
del allLabes
del integerLables
del allImagesForModel

# build the model

from tensorflow.keras.layers import Dense , Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.nasnet import NASNetLarge

myModel = NASNetLarge(input_shape=IMAGE_FULL_SIZE , weights='imagenet', include_top=False)

# we dont want to train the existing layers
for layer in myModel.layers:
    layer.trainable = False
    print(layer.name)

#add Flatten layer
plusFlattenLayer = Flatten()(myModel.output)

#add the last dense layer with out 120 classes 
predicition = Dense(numOfCategories, activation='softmax')(plusFlattenLayer)

model = Model(inputs=myModel.input, outputs=predicition)

#print(model.summary())

from tensorflow.keras.optimizers import Adam

lr = 1e-4 # 0.0001
opt = Adam(lr)

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = opt,
    metrics=['accuracy'] )

stepsPerEpoch = np.ceil(len(X_train) / batchSize)
validationSteps = np.ceil(len(X_test) / batchSize)


# early stopping 
from keras.callbacks import ModelCheckpoint , ReduceLROnPlateau , EarlyStopping

best_model_file = "e:/temp/dogs.h5"

callbacks = [
        ModelCheckpoint(best_model_file, verbose=1 , save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1 , verbose=1, min_lr=1e-6),
        EarlyStopping(monitor='val_accuracy', patience=7, verbose=1) ]


# train the model (fit)

r = model.fit (
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs = 30,
    batch_size = batchSize,
    steps_per_epoch=stepsPerEpoch,
    validation_steps=validationSteps,
    callbacks=[callbacks]
)










