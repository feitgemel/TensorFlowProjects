# relelvant Python Libraries :
# pip install tensorflow
# pip install xgboost
# pip install opencv-python
# pip install numpy

# dataset: 
#load dataset : https://www.kaggle.com/datasets/mrtontrnok/5-vehichles-for-multicategory-classification

import numpy as np
import os
import cv2
import glob

print(os.listdir("E:/Data-sets/5 vehichles for classification/"))

SIZE = 256


# load the train data
train_images = []
train_labels = []

for direcotory_path in glob.glob("E:/Data-sets/5 vehichles for classification/train/*"):
    label = direcotory_path.split("\\")[-1]

    #print(label)

    for img_path in glob.glob(os.path.join(direcotory_path,"*.png")):
        #print(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE,SIZE))

        train_images.append(img)
        train_labels.append(label)

train_imagesNP = np.array(train_images)
train_labelsNP = np.array(train_labels)

print(train_labelsNP)
print(train_imagesNP.shape)
print(train_labelsNP.shape)

# load the validation data 

validation_images = []
validation_labels = []

for direcotory_path in glob.glob("E:/Data-sets/5 vehichles for classification/validation/*"):
    label = direcotory_path.split("\\")[-1]

    print(label)

    for img_path in glob.glob(os.path.join(direcotory_path,"*.png")):
        print(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE,SIZE))

        validation_images.append(img)
        validation_labels.append(label)

validation_imagessNP = np.array(validation_images)
validation_labelsNP = np.array(validation_labels)

print(validation_labelsNP)
print(validation_imagessNP.shape)
print(validation_labelsNP.shape)


print("Save the data :")
np.save("e:/temp/5-vehicles-train-images.npy",train_imagesNP)
np.save("e:/temp/5-vehicles-train-labels.npy",train_labelsNP)
np.save("e:/temp/5-vehicles-validate-images.npy",validation_imagessNP)
np.save("e:/temp/5-vehicles-validate-labels.npy",validation_labelsNP)
print("Finish save the data .............")

