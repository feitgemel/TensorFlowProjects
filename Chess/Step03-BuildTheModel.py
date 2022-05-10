# Create the CNN model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint
import matplotlib.pyplot as plt
from glob import glob

imgWidth = 256
imgHeight = 256
batchSize = 32
numOfEpochs = 100

TRAINING_DIR = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/train"

NumOfClasses = len(glob('C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/train/*')) # dont forget the '  /*  '
print (NumOfClasses) # 6 classes

# data augmentation to increase the train data

train_datagen = ImageDataGenerator(rescale = 1/255.0, #normalize between 0 - 1
                                    rotation_range = 30 ,
                                    zoom_range = 0.4 ,
                                    horizontal_flip=True,
                                    shear_range=0.4)


train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size = batchSize,
                                                    class_mode = 'categorical',
                                                    target_size = (imgHeight,imgWidth))


validation_DIR = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/validation"
val_datagen = ImageDataGenerator(rescale = 1/255.0)

val_generator = val_datagen.flow_from_directory(validation_DIR,
                                                batch_size = batchSize,
                                                class_mode='categorical',
                                                target_size = (imgHeight, imgWidth))


# early stopping

callBack = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')


# if we will find a better model we will save it here :
bestModelFileName = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/chess_best_model.h5"

bestModel = ModelCheckpoint(bestModelFileName, monitor='val_accuracy', verbose=1, save_best_only=True)

# the model :

model = Sequential([ 
    Conv2D(32, (3,3) , activation='relu' , input_shape=(imgHeight, imgWidth, 3) ) ,
    MaxPooling2D(2,2),
    
    Conv2D(64 , (3,3) , activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64 , (3,3) , activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128 , (3,3) , activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(256 , (3,3) , activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(512 , activation='relu'),
    Dense(512 , activation='relu'),

    Dense(NumOfClasses , activation='softmax') # softmax -> 0 to 1 
])

print (model.summary() )


# compile the model with Adam optimizer

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator,
                    epochs = numOfEpochs,
                    verbose=1,
                    validation_data = val_generator,
                    callbacks = [bestModel])


# display the result using pyplot

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc)) # for the max value in the diagram


# accuracy chart

fig = plt.figure(figsize=(14,7))
plt.plot(epochs, acc , 'r', label="Train accuracy")
plt.plot(epochs, val_acc , 'b', label="Validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and validation accuracy')
plt.legend(loc='lower right')
plt.show()

#loss chart
fig2 = plt.figure(figsize=(14,7))
plt.plot(epochs, loss , 'r', label="Train loss")
plt.plot(epochs, val_loss , 'b', label="Validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and validation Loss')
plt.legend(loc='upper right')
plt.show()

