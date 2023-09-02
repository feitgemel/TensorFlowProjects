# In this tutorial we will make a fruit and vegtables classificaion model using TensorFlow and Keras
# first , lets look for a dataset :
# https://www.kaggle.com/kritikseth/fruit-and-vegetable-image-recognition

import os
import cv2
import numpy as np
from numpy import save

# Array of all the classes 

class_names = ["banana", "apple", "pear", "grapes", "orange", "kiwi", "watermelon", "pomegranate", "pineapple", "mango", "cucumber", "carrot", "capsicum", "onion", "potato", "lemon", "tomato", "raddish", "beetroot", "cabbage", "lettuce", "spinach", "soy beans", "cauliflower", "bell pepper", "chilli pepper", "turnip", "corn", "sweetcorn", "sweetpotato", "paprika", "jalepeno", "ginger", "garlic", "peas", "eggplant"]

train_data_array = []
train_data_labels_array = []

print ("Loading the train data ")

rootdir = "C:/Python-cannot-upload-to-GitHub/Fruit-and-Vegetable/train"

for subdir , dirs , files in os.walk(rootdir):
    for file in files:
        # lets open each image in the folders
        frame = cv2.imread(os.path.join(subdir, file))

        # check the validity of the image 
        if frame is None:
            print("not an image")
        else:
            print(subdir,file)

            # lets check the sizes of the images to see if all are the same
            # we can see that each image has its won size and dimetions
            # so , We have to resized all the images to the same size
            # We will reduce the size of the images to 28X28
            resized = cv2.resize(frame,(28,28), interpolation=cv2.INTER_AREA)
            checkSize = resized.shape[0] #checking that the resize was done successfuly
            if checkSize ==28 :
                train_data_array.append(resized)
                index = class_names.index(os.path.basename(subdir))
                train_data_labels_array.append(index)

# converts the lists to numpy arrays
train_data = np.array(train_data_array)
train_data_lables = np.array(train_data_labels_array)

print ("Finished loading the train data")
print ("Number of train records : ", train_data.shape[0] )

print(train_data.shape)
print(train_data_lables.shape)

# while running , lets add more code :

# lets see 2 examples of the images :

# demoImage = train_data[4] # lets look at image number 4
# cv2.imshow("Demo image", demoImage)
# index = train_data_lables[4]
# print (class_names[index])

# #lets add another sample image 
# demoImage2 = train_data[5] # lets look at image number 4
# cv2.imshow("Demo image2", demoImage2)
# index = train_data_lables[5]
# print (class_names[index])

# cv2.waitKey(0)


# lets save the data to the disk in numpy binary format:

save('c:/temp/train_data.npy', train_data)
save('c:/temp/train_data_labels.npy', train_data_lables)


#lets continue to the test data 
print("Start loading the test data ")
rootdir = "C:/Python-cannot-upload-to-GitHub/Fruit-and-Vegetable/test"

test_data_array = []
test_data_labels_array = []

# lets build another arrray for the bigger images , so we can see it after building the Tensorflow model
test_data_big_array = []


for subdir , dirs , files in os.walk(rootdir):
    for file in files:
        # lets open each image in the folders
        frame = cv2.imread(os.path.join(subdir, file))

        # check the validity of the image 
        if frame is None:
            print("not an image")
        else:
            print(subdir,file)
            # just to have the ability to see a "normal" size image
            resizedBig = resized = cv2.resize(frame,(280,280), interpolation=cv2.INTER_AREA)
            resized = cv2.resize(frame,(28,28), interpolation=cv2.INTER_AREA)
            checkSize = resized.shape[0] #checking that the resize was done successfuly
            if checkSize ==28 :
                test_data_array.append(resized)
                test_data_big_array.append(resizedBig)
                index = class_names.index(os.path.basename(subdir))
                test_data_labels_array.append(index)

test_data = np.array(test_data_array)
test_data_big = np.array(test_data_big_array)
test_data_labels = np.array(test_data_labels_array)

print("Finished loading the test data ")
print ("Number of test records : ", test_data.shape[0] )
print(test_data.shape)
print(test_data_labels.shape)

# save the numpy arrays as numpy binary to the disk 
save('c:/temp/test_data.npy', test_data)
save('c:/temp/test_data_big.npy', test_data_big)
save('c:/temp/test_data_labels.npy', test_data_labels)


