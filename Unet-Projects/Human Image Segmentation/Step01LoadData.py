# dataset : https://www.kaggle.com/datasets/nikhilroxtomar/person-segmentation

import cv2
import numpy as np
import pandas as pd

Height = 256
Width = 256

allImages = []
maskImages = []

allValidateImages = []
maskValidatImages = []

path = "e:/Data-sets/people_segmentation/"
imagesPath = path + "images"
maskPath = path + "masks"

TrainFile = path + "segmentation/train.txt"
validateFile = path + "segmentation/val.txt"

# train Data
df = pd.read_csv(TrainFile, sep=" ", header=None)
filesList = df[0].values

#print(filesList)

# load one image and one mask

# image
img = cv2.imread(imagesPath+"/ache-adult-depression-expression-41253.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(img, (Width, Height))
cv2.imshow("img", img)

#mask
mask = cv2.imread(maskPath+"/ache-adult-depression-expression-41253.png", cv2.IMREAD_GRAYSCALE)
MASK16 = cv2.resize(mask , (16,16))
print(MASK16)

# 0 is the background
# 1 is the human

# lets multiple the values with 255 -> to generate an image
mask = cv2.resize(mask , (Width, Height))
mask = mask * 255
cv2.imshow("mask", mask)

cv2.waitKey(0)

# load all the train images and masks
# ==================================

print("Start loading the train images and masks ..............................")
for file in filesList:
    filePathForImage = imagesPath + "/" +file + ".jpg"
    filePathForMask = maskPath + "/" + file + ".png"

    print(file)

    img = cv2.imread(filePathForImage , cv2.IMREAD_COLOR)
    img = cv2.resize(img , (Width, Height))
    img = img / 255.0
    img = img.astype(np.float32)
    allImages.append(img)



    mask = cv2.imread(filePathForMask, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask , (Width, Height))
    maskImages.append(mask)


allImagesNP = np.array(allImages)
maskImagesNP = np.array(maskImages)
maskImagesNP = maskImagesNP.astype(int) # convert the values to integers


print ("Shapes of train images and masks :")
print(allImagesNP.shape)
print(maskImagesNP.shape)
print(maskImagesNP.dtype)


# load the Validate images and masks
# ==================================
df = pd.read_csv(validateFile, sep=" ", header=None)
filesList = df[0].values

print("Start loading the Validate images and masks ..............................")
for file in filesList:
    filePathForImage = imagesPath + "/" +file + ".jpg"
    filePathForMask = maskPath + "/" + file + ".png"

    print(file)

    img = cv2.imread(filePathForImage , cv2.IMREAD_COLOR)
    img = cv2.resize(img , (Width, Height))
    img = img / 255.0
    img = img.astype(np.float32)
    allValidateImages.append(img)



    mask = cv2.imread(filePathForMask, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask , (Width, Height))
    maskValidatImages.append(mask)


allValidateImagesNP = np.array(allValidateImages)
maskValidateImagesNP = np.array(maskValidatImages)
maskValidateImagesNP = maskValidateImagesNP.astype(int) # convert the values to integers


print ("Shapes of train images and masks :")
print(allValidateImagesNP.shape)
print(maskValidateImagesNP.shape)
print(maskValidateImagesNP.dtype)

print("Save the Data ......")

np.save("e:/temp/Unet-Human-Train-Images.npy", allImagesNP)
np.save("e:/temp/Unet-Human-Train-masks.npy", maskImagesNP)
np.save("e:/temp/Unet-Human-Validate-Images.npy", allValidateImagesNP)
np.save("e:/temp/Unet-Human-Validate-Masks.npy", maskValidateImagesNP)

print("Finish save the data .............")
