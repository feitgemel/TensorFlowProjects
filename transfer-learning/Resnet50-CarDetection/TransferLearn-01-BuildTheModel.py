from tensorflow.keras.layers import Input , Lambda , Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50 , preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator , load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob 
import matplotlib.pyplot as plt 


IMAGE_SIZE = [224,224]

TrainFolder = "C:/Python-cannot-upload-to-GitHub/Cars/Train"
ValidateFolder = "C:/Python-cannot-upload-to-GitHub/Cars/Validate"

myResnet = ResNet50(input_shape= IMAGE_SIZE+[3] , weights='imagenet', include_top=False ) # include_top=False ==> remove the fully connected layer

print ( myResnet.summary() )

for layer in myResnet.layers:
    layer.trainable = False # we dont need to train the model . It is already trained

# classes 
Classes = glob('C:/Python-cannot-upload-to-GitHub/Cars/Train/*') #dont forget the /*
print(Classes)

classesNum = len(Classes)
print(classesNum)

# continue with the next layers of the model :

# add Flatten layer 
PlusFlattenlayer = Flatten()(myResnet.output)

# add a Dense layer with our classes 
prediction = Dense(classesNum, activation='softmax')(PlusFlattenlayer)


# create the model amd add tje mew layers 

model = Model(inputs=myResnet.input , outputs=prediction)

print (model.summary())         

# compile the model

model.compile(
    loss = 'categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])


# images augmentaion

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale= 1. /255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. /255)

training_set = train_datagen.flow_from_directory(TrainFolder, target_size=(224,224),batch_size=32,class_mode='categorical')
test_set = test_datagen.flow_from_directory(ValidateFolder,target_size=(224,224),batch_size=32,class_mode='categorical')

# fit the model
result = model.fit(training_set ,
    validation_data=test_set,
    epochs=50,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set))

# plot the result

# plot the accuracy
plt.plot(result.history['accuracy'],label='train_acc')
plt.plot(result.history['val_accuracy'],label='val_acc')
plt.legend()
plt.show()

#plot the loss
plt.plot(result.history['loss'],label='train_loss')
plt.plot(result.history['val_loss'],label='val_loss')
plt.legend()
plt.show()

# save the model
model.save('C:/Python-cannot-upload-to-GitHub/Cars/myCarsModel.h5')






