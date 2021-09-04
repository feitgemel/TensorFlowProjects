# Now we will move to next step of building a model using Tensor Flow and Keras

import os
import cv2
import numpy as np
from numpy import load
from numpy.core.defchararray import index


import tensorflow as tf 
from tensorflow import keras
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2" # Reduce the information messages

# Array of all the classes 
class_names = ["banana", "apple", "pear", "grapes", "orange", "kiwi", "watermelon", "pomegranate", "pineapple", "mango", "cucumber", "carrot", "capsicum", "onion", "potato", "lemon", "tomato", "raddish", "beetroot", "cabbage", "lettuce", "spinach", "soy beans", "cauliflower", "bell pepper", "chilli pepper", "turnip", "corn", "sweetcorn", "sweetpotato", "paprika", "jalepeno", "ginger", "garlic", "peas", "eggplant"]

#load the saved train and test data
train_data = load('c:/temp/train_data.npy')
train_data_lables = load('c:/temp/train_data_labels.npy')
test_data = load('c:/temp/test_data.npy')
test_data_big = load('c:/temp/test_data_big.npy')
test_data_labels = load('c:/temp/test_data_labels.npy')

print("Finish loading the data ")

#show a sample data - image number 116 in the train data
# demoImage = train_data[116]
# cv2.imshow('demoImage',demoImage)
# index = train_data_lables[116]
# print(class_names[index])
# cv2.waitKey(0)

# data shape :

print("train shape : ", train_data.shape)
print("train lables shape : ", train_data_lables.shape)
print("test data shape:", test_data.shape)
print("test data labels shape:", test_data_labels.shape)

# we have 3579 images to train in a 28X28 resolution with 3 channels (RGB)
# After the train we will test the images with 359 test data

# the values of each pixel is 0 to 255 . We would like ot change it between 0 to 1

train_data = train_data / 255.0
test_data = test_data / 255.0

# build the model

model = keras.Sequential([
    # first we will flatten the images . We will take the 28X28X3 and flatten the shape as the input to the model
    keras.layers.Flatten(input_shape=(28,28,3)), # this is the input layer

    # lets define the hidden layer.
    # we dont know what is the exact number , so will try with 512 neurons
    keras.layers.Dense(512,activation='relu'), # relu has no negative values.

    # this is the last layer - the classifation for the classes 
    # we have 36 classes (Apple , banana , orange .........)
    keras.layers.Dense(36,activation='softmax') # softmax has return values between 0 to 1 
])

print('Finish build the model skeleton')

# compile the model
model.compile(
    # optimizer -> calulate the gradient descent of the network
    optimizer='adam',
    # loss function
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy'] # metrics measurment
)

print('Finish compile the model')

# train the model
# we start with a 10 epochs , but probebly will update the number to a bigger one during the sessions
model.fit(train_data,train_data_lables,epochs=120)

# we got loss: 0.2574 - accuracy: 0.9329
#lets test it on a new data . A data that the model never seen

test_loss , test_acc = model.evaluate(test_data,test_data_labels,verbose=1) # verbose is a paramter of how detailed is the log in the console
print("*******************         Test accuracy : ", test_acc)

# we have a result of : Test accuracy :  0.941504180431366  ## very good result - near 1

# predictions
predictions = model.predict(test_data)
#print(predictions) # we will get predicitions for the whole test data

# lets show the predictions of a specific image , for example : image number 100
print ('The predicted class index :')

#first we will show the outcome 
# for every test image we will get 36 numbers between 0 to 1
# we got 36 numbers.
# the higher number is the predicted class , so we have to extracr the index in the 36 list

class_index = np.argmax(predictions[100]) # get the max value
print(class_index)

class_name = class_names[class_index]
print('the class name :', class_name)

# lets show the image number 100
# We saved test data as bigger files 280X280 . We will use it now

demoImage = test_data_big[100]
cv2.putText(demoImage,class_name,(20,20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),1)
cv2.imshow('demoImage',demoImage)
cv2.waitKey(0)

# lets compare all the results :

for predict , test_label in zip(predictions,test_data_labels):
    class_index = np.argmax(predict)
    class_name_predict = class_names[class_index]

    class_name_original = class_names[test_label]

    print('Predicted class :',class_name_predict , '     Original / real class name :', class_name_original )


# you can see that the preictions are very well !!!!
# Thank you , and bye bye
