from re import I
from keras.layers import Dense, Flatten
from keras.models import Model

# vgg19
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

import numpy as np
import matplotlib.pyplot as plt
from glob import glob

IMAGE_SIZE = [150,150]

trainImagesFolder = "C:/Python-cannot-upload-to-GitHub/Weather/weather-data/train"
validationImagesFolder = "C:/Python-cannot-upload-to-GitHub/Weather/weather-data/validation"

# data augmentation

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                    shear_range = 0.4,
                                    zoom_range= 0.4,
                                    rotation_range=0.4,
                                    horizontal_flip= True)


valid_datagen = ImageDataGenerator( rescale= 1. / 255)


train_data_set = train_datagen.flow_from_directory(trainImagesFolder,
                                                    target_size=(150,150),
                                                    batch_size=32,
                                                    class_mode='categorical')

                                                    
valid_data_set = valid_datagen.flow_from_directory(validationImagesFolder,
                                                    target_size=(150,150),
                                                    batch_size=32,
                                                    class_mode='categorical')

# 1274 - train images
#226 - valid images

myVgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False) 
# include_top=False -> remove the last fully connected layes from the VGG19 , so we can add our own layers

for layer in myVgg.layers:
    layer.trainable = False


Classes = glob('C:/Python-cannot-upload-to-GitHub/Weather/weather-data/train/*')
print(Classes)

classesNum = len(Classes)
print ('Number of Classes : ')
print(classesNum)

# lets add the ajustments to the VGG19 model

model = Sequential()
model.add(myVgg)
model.add(Flatten())
model.add(Dense(classesNum , activation='softmax'))

print (model.summary())


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint , EarlyStopping
checkpoint = ModelCheckpoint('C:/Python-cannot-upload-to-GitHub/Weather/weather-data/MyVgg19Option2.h5', 
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True)

earlystop = EarlyStopping(monitor='val_accuracy', patience=5 , verbose=1)                             

# fit the model (Training)

result = model.fit(train_data_set, validation_data=valid_data_set , epochs=15, verbose=1 , callbacks=[checkpoint,earlystop])

# plot accuracy 
plt.plot(result.history['accuracy'], label='train accuracy')
plt.plot(result.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()

# plot loss 
plt.plot(result.history['loss'], label='train loss')
plt.plot(result.history['val_loss'], label='val loss')
plt.legend()
plt.show()



