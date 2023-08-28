# The Oxford-IIIT Pet Dataset :
# https://www.robots.ox.ac.uk/~vgg/data/pets/

import pandas as pd
import cv2
import numpy as np

Height = 128
Width= 128
NumOfCategories = 3

#The Mask images has 3 type of data (Expain later )
# Value = 1. indicates the main object (The animal)
# value = 2. Indicates the background
# value = 3. Indicates the border of the object (The animal)


# Create arrayes for the train and test (images and masks)
# train
allImages = []
maskImages = []

#test :
allTestImages = []
maskTestImages = []

path = "E:/Data-sets/Unet-Multi-class/"
trainFile = path + "annotations/trainval.txt"
testFile = path + "annotations/test.txt"

# load the train images and the original masks
print("Load train data : ")

df = pd.read_csv(trainFile, sep=" ", header=None)
names = df[0].values # get the list of files
print ("Train data info :")
print(len(names))

# load the images

for name in names :
    imageFileName = path + "images/" + name + ".jpg"
    print(imageFileName)

    img = cv2.imread(imageFileName , cv2.IMREAD_COLOR)
    img = cv2.resize(img, (Width,Height))
    img = img / 255.0
    img = img.astype(np.float32)
    allImages.append(img)

    # mask
    maskFileName = path + "annotations/trimaps/" + name + ".png"
    mask = cv2.imread(maskFileName , cv2.IMREAD_GRAYSCALE) # gray scale images
    mask = cv2.resize(mask , (Width, Height))
    maskImages.append(mask)

#print(len(allImages))
#print(len(maskImages))

allImagesNP = np.array(allImages)
maskImagesNP = np.array(maskImages)
maskImagesNP = maskImagesNP.astype(int)

print(allImagesNP.shape)
print(allImagesNP.dtype)

print(maskImagesNP.shape)
print(maskImagesNP.dtype)

# lets display and learn about the mask
# lets load one image forn the numpy array and print it in a reduced size

x = cv2.resize(maskImagesNP[0], (16,16), interpolation=cv2.INTER_NEAREST)
print(x)

# value = 1 - main object
# value = 2 background
# value = 3 border of the object

# we have to replace tha values between 0 -> 2

# loop each row in the array 

for i in range(len(x)):

    # loop through each element in the row
    for j in range(len(x[i])):

        # replace the values
        v = x[i][j]

        if v==1 : # the object
            x[i][j] = 0
        if v==2 :
            x[i][j] = 22 # the background

        if v==3 :
            x[i][j] = 333

# show the result
print(x)  

# load the test images

print("load test data :")

df = pd.read_csv(testFile, sep=" ", header=None)
names = df[0].values # get the list of files

print ("Test data info :")
print(len(names))

# load the images

for name in names :
    imageFileName = path + "images/" + name + ".jpg"
    print(imageFileName)

    img = cv2.imread(imageFileName , cv2.IMREAD_COLOR)
    img = cv2.resize(img, (Width,Height))
    img = img / 255.0
    img = img.astype(np.float32)
    allTestImages.append(img)

    # mask
    maskFileName = path + "annotations/trimaps/" + name + ".png"
    mask = cv2.imread(maskFileName , cv2.IMREAD_GRAYSCALE) # gray scale images
    mask = cv2.resize(mask , (Width, Height))
    maskTestImages.append(mask)

#print(len(allImages))
#print(len(maskImages))

allTestImagesNP = np.array(allTestImages)
maskTestImagesNP = np.array(maskTestImages)
maskTestImagesNP = maskTestImagesNP.astype(int)

print(allTestImagesNP.shape)
print(allTestImagesNP.dtype)

print(maskTestImagesNP.shape)
print(maskTestImagesNP.dtype)

print("Save the Data :")
np.save("e:/temp/Unet-Animals-train-images.npy", allImagesNP)
np.save("e:/temp/Unet-Animals-train-mask.npy", maskImagesNP)
np.save("e:/temp/Unet-Animals-test-images.npy", allTestImagesNP)
np.save("e:/temp/Unet-Animals-test-mask.npy", maskTestImagesNP)
print("Finish save the data !")
