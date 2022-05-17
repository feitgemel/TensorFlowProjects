from keras.layers import Dense , Flatten
from keras.models import Model

# vgg model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

from keras.preprocessing import image

#data augmentation
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# lets see our images
trainImagesFolder = "C:/Python-cannot-upload-to-GitHub/MyImages/Train"
validImagesFolder = "C:/Python-cannot-upload-to-GitHub/MyImages/validate"

IMAGE_SIZE = [224,224]

myVgg = VGG16(input_shape=IMAGE_SIZE + [3] , weights='imagenet', include_top=False)
print (myVgg.summary())

for layer in myVgg.layers:
    layer.trainable = False

classesNum = 3

# add flatten layer
PlusFlattenLayer = Flatten()(myVgg.output)
predicitionLayer = Dense(classesNum , activation='softmax')(PlusFlattenLayer) 

model = Model(inputs=myVgg.input , outputs=predicitionLayer)

print( model.summary())

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])




# Image augmentaion

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range=0.2 ,
                                   zoom_range = 0.2 ,
                                   horizontal_flip=True )

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory(trainImagesFolder,
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode='categorical')

validation_set = train_datagen.flow_from_directory(validImagesFolder,
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode='categorical')



result = model.fit(training_set,
                    validation_data=validation_set,
                    epochs=10,
                    steps_per_epoch=len(training_set),
                    validation_steps=len(validation_set) )



# plot the accuracy

plt.plot(result.history['accuracy'], label='train acc')
plt.plot(result.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

# plot the loss

plt.plot(result.history['loss'], label='train loss')
plt.plot(result.history['val_loss'], label='val loss')
plt.legend()
plt.show()

#save the model 
model.save("C:/Python-cannot-upload-to-GitHub/MyImages/myFaceModel.h5")