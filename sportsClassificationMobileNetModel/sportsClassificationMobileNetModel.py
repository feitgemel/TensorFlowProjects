# in this project we will train sports images in order to classifiy the sports type
# Then we will predict the sports type by showing a new image to the model
# we are going to use the Mobilnet model as a start , and we will twist the "tail" of the neural network
# in order to ajust it to our custom target.

# first , lets download a sports data set
# we are going to use the 73 classes images from Kaggle

# as you can see , this data set has 73 classes .
# We will try to build a model that can learn the images and predict the sports type .
# just for this lesson , we will use only 21 of the classes , but you can try to train more .
# It is only a matter of time and the amounts of ephocs.

# now , I will stop the video to build the directory
# please notice that everything is basicly ready since the dataset is already arranged as Train , test and validate

# After copying the images
# We have 3 directoies , Each directory has 21 classess .

# this tutorial is based that you already install Tensor Flow , and OpenCV as well

from scipy.ndimage.measurements import label
import tensorflow as tf
import os
import keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.python.keras.engine.training import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt 
import numpy as np
import cv2
from tensorflow.python.ops.gradients_util import _Inputs


# check if we have a GPU
physicalDevices = tf.config.experimental.list_physical_devices('GPU')
print ('Num GPUs available : ', len(physicalDevices))
tf.config.experimental.set_memory_growth(physicalDevices[0], True)
print ('============================================')

#lets test this basic import code 
# I have one GPU. But you can run it with CPU as well . (only a matter of time )

# lets load the data to a train set , validate set and test set

# lets change the working directory .

os.chdir('C:/SportsImages')

# the folders inside this directory are :

train_path = 'train'
valid_path = 'valid'
test_path = 'test'

#inside each folder we have the 21 classes
# lets build a array of classes  

class_names = ["air hockey",
"ampute football",
"archery",
"arm wrestling",
"balance beam",
"barell racing",
"baseball",
"basketball",
"billiards",
"bmx",
'swimming',
'table tennis',
'tennis',
'track bicycle',
'tug of war',
'uneven bars',
'volleyball',
'water polo',
'weightlifting',
'wheelchair basketball',
'wheelchair racing',
]

# we are going to use the mobile net model
# we have to convert the images to the requirements of the model
# the model also include the target classs .
# the mobile net model works in batches . We will define 10 images in each batch

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224),classes=class_names,batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=valid_path, target_size=(224,224),classes=class_names,batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224),classes=class_names,batch_size=10, shuffle=False)

#each object will have the data : train , validate and test.

# lets grabe one batch (= 10 images) , and show them

imgs, labels = next(train_batches)

# lets use a function that show one batch (10 images)
def plotImages(images_arr) :
    flg , axes = plt.subplots(1,10,figsize= (20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# lets call the function :) 
#plotImages(imgs)

# important - all the data set are suffuled . we would like the test data will not be shuffuled . You will see it later

# please look :
#Train - Found 2844 images belonging to 21 classes.
#Validate - Found 105 images belonging to 21 classes.
#Test - Found 80 images belonging to 21 classes

# lets load the model
mobile = tf.keras.applications.mobilenet.MobileNet()

# lets see the archituecture of the model and the layers

#mobile.summary()

# we are going to use all the layers from the begining untill 6 layers to last
# we will replace the 6 last layers with our Dense layer

x = mobile.layers[-6].output

# Now , we will create the last layer -> named it output


# units 21 - since we have 21 classes to predict
# softmax , since we need an activation layer between 0 to 1
output = Dense(units=21, activation='softmax')(x)

# create the merged model

model = Model(inputs=mobile.input , outputs = output)

# now - we will freeze all execpt the last 23 layers (out of 88 original model)
for layer in model.layers[:-23]:
    layer.trainable = False


# lets see our new model
model.summary()

# dense (Dense)                (None, 21)                21525 -> our layer 

# lets train the model

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# run the model with 30 epochs 

model.fit(x=train_batches, validation_data=valid_batches, epochs=10 , verbose=2)

#285/285 - 8s - loss: 0.0048 - accuracy: 0.9989 - val_loss: 0.0167 - val_accuracy: 0.9905
# the results seems very good. lets see one image for example

# lets grab the classes
test_lables = test_batches.classes
print (test_lables)

predictions = model.predict(x=test_batches , verbose=0)

# lets see image number 6
print ( predictions[6] )
class_index = np.argmax(predictions[6]) # get the position of the largest value
class_name_predicted = class_names[class_index] 
class_name_original = class_names[ test_lables[6] ]


print('predict class 6', class_name_predicted)
print('original class 6',class_name_original)


imgs, lables = next(test_batches) # get the first 10 images 

img = imgs[6]
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

img = cv2.putText(img,class_name_predicted,(5,55),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
cv2.imshow('img',img)
cv2.waitKey(0)

# lets see more random predictions
imgs, lables = next(test_batches) # get the next 10 images 
imgs, lables = next(test_batches) # get the next 10 images 

class_index = np.argmax(predictions[25]) # get the position of the largest value
class_name_predicted = class_names[class_index] 

img = imgs[5] # since it is the fifth image in the third batch -> aim to the 25 image
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

img = cv2.putText(img,class_name_predicted,(5,55),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.imwrite('test2.jpg',img)

# lets test another one :
imgs, lables = next(test_batches) # get the next 10 images 

class_index = np.argmax(predictions[35]) # get the position of the largest value
class_name_predicted = class_names[class_index] 

img = imgs[5] # since it is the fifth image in the third batch -> aim to the 25 image
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

img = cv2.putText(img,class_name_predicted,(5,55),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
cv2.imshow('img',img)
cv2.waitKey(0)

cv2.imwrite('test3.jpg',img)













