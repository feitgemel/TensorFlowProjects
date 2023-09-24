# Search in Google : "ISIC Challenge Datasets 2018" and go to 2018 tab
# Dataset : https://challenge.isic-archive.com/data/#2018
# 2594 images and 12970 corresponding ground truth response masks (5 for each image)
# 10.5 Giga

import cv2
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

Height = 128
Width = 128

path = "E:/Data-sets/Melanoma/"
imagesPath = path + "ISIC2018_Task1-2_Training_Input/*.jpg"
maskPath = path + "ISIC2018_Task1_Training_GroundTruth/*.png"

print("Images in folder :")
listofImages = glob.glob(imagesPath)
print(len(listofImages))

listOfMasks = glob.glob(maskPath)
print(len(listOfMasks))


#load one image and one mask

img = cv2.imread(listofImages[0], cv2.IMREAD_COLOR)
print(img.shape)
img = cv2.resize(img, (Width, Height))

mask = cv2.imread(listOfMasks[0], cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask, (Width, Height))


# augmentation - Create a smaple augmentation

import imgaug as ia
import imgaug.augmenters as iaa

hflip = iaa.Fliplr(p=1.0)
hflipImg = hflip.augment_image(img)

vflip= iaa.Flipud(p=1.0)
vflipImg = vflip.augment_image(img)

rot1 = iaa.Affine(rotate=(-50,20))
rotImg = rot1.augment_image(img)

cv2.imshow("hflipImg",hflipImg)
cv2.imshow("vflipImg",vflipImg)
cv2.imshow("rotImg",rotImg)




# invetigate the mask
# downsize the mask , so we can look at the values :

mask16 = cv2.resize(mask , (16,16))
print(mask16)

# 0 -> background (black)
# 255 -> is the object (white object)
####  We prefer values of 0 and 1

mask16[mask16 > 0]=1

print("==================================  New mask ")
print(mask16)

cv2.imshow("img",img)
cv2.imshow("mask", mask)

cv2.waitKey(0)


# create the full data

allImages = []
maskImages = []

print(len(listofImages))
print(len(listOfMasks))  


# load all trained images and masks
print("Start loading the train images and masks , and augment each image in 3 types ")

for imgFile , imgMask in tqdm(zip(listofImages, listOfMasks), total=len(listofImages)) :

    # create numpy arrays for the images
    img = cv2.imread(imgFile , cv2.IMREAD_COLOR)
    img = cv2.resize(img , (Width,Height))
    img = img / 255.0
    img = img.astype(np.float32)
    allImages.append(img)

    mask = cv2.imread(imgMask, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask , (Width, Height))
    mask[mask > 0]= 1
    maskImages.append(mask)

    # data augmentation
    hflip = iaa.Fliplr(p=1.0)
    hflipImg = hflip.augment_image(img)
    hflipMask = hflip.augment_image(mask)
    allImages.append(hflipImg)
    maskImages.append(hflipMask)

    vlip= iaa.Flipud(p=1.0)
    vflipImg = vflip.augment_image(img)
    vflipMask = vflip.augment_image(mask)
    allImages.append(vflipImg)
    maskImages.append(vflipMask)

    rot1 = iaa.Affine(rotate=(-50,20))
    rotImg = rot1.augment_image(img)
    rotMask = rot1.augment_image(mask)
    allImages.append(rotImg)
    maskImages.append(rotMask)



allImagesNP = np.array(allImages)
maskImagesNP = np.array(maskImages)
maskImagesNP = maskImagesNP.astype(int) # convert the values to Interger

print("Shapes of train Images and masks")
print(allImagesNP.shape)
print(maskImagesNP.shape)


# Load the test images 

imagesPath = path + "ISIC2018_Task1-2_Test_Input/*.jpg"
maskPath = path + "ISIC2018_Task1_Test_GroundTruth/*.png"

allTestImages = []
maskTestImages = []

print("Images in folder :")
listofImages = glob.glob(imagesPath)
print(len(listofImages))

listOfMasks = glob.glob(maskPath)
print(len(listOfMasks))

print("Start loading the Test(!!!) images and masks , and augment each image in 3 types ")

for imgFile , imgMask in tqdm(zip(listofImages, listOfMasks), total=len(listofImages)) :

    # create numpy arrays for the images
    img = cv2.imread(imgFile , cv2.IMREAD_COLOR)
    img = cv2.resize(img , (Width,Height))
    img = img / 255.0
    img = img.astype(np.float32)
    allTestImages.append(img)

    mask = cv2.imread(imgMask, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask , (Width, Height))
    mask[mask > 0]= 1
    maskTestImages.append(mask)

    # data augmentation
    hflip = iaa.Fliplr(p=1.0)
    hflipImg = hflip.augment_image(img)
    hflipMask = hflip.augment_image(mask)
    allTestImages.append(hflipImg)
    maskTestImages.append(hflipMask)

    vlip= iaa.Flipud(p=1.0)
    vflipImg = vflip.augment_image(img)
    vflipMask = vflip.augment_image(mask)
    allTestImages.append(vflipImg)
    maskTestImages.append(vflipMask)

    rot1 = iaa.Affine(rotate=(-50,20))
    rotImg = rot1.augment_image(img)
    rotMask = rot1.augment_image(mask)
    allTestImages.append(rotImg)
    maskTestImages.append(rotMask)



allTestImagesNP = np.array(allTestImages)
maskTestImagesNP = np.array(maskTestImages)
maskTestImagesNP = maskTestImagesNP.astype(int) # convert the values to Interger

print("Shapes of train Images and masks")
print(allTestImagesNP.shape)
print(maskTestImagesNP.shape)


# save the Data :

print("Save the Train Data :")
np.save("e:/temp/Unet-Train-Melanoa-Images.npy", allImagesNP)
np.save("e:/temp/Unet-Train-Melanoa-Masks.npy", maskImagesNP)

print("Save the Test Data : ")
np.save("e:/temp/Unet-Test-Melanoa-Images.npy", allTestImagesNP)
np.save("e:/temp/Unet-Test-Melanoa-Masks.npy", maskTestImagesNP)

print("Finish save the Data ..........................")


