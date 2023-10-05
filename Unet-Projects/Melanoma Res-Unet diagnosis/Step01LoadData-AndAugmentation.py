#Dataset : https://challenge.isic-archive.com/data/#2018
# 2594 images and 12970 corresponding ground truth response masks (5 for each image)
# 10.5 Giga

#Dataset : https://challenge.isic-archive.com/data/#2018
# 2594 images and 12970 corresponding ground truth response masks (5 for each image)
# 10.5 Giga

# https://arxiv.org/pdf/1711.10684.pdf

#ResNet uses a skip connection in which an original input is also added to the output of the convolution block.
# This helps in solving the problem of vanishing gradient by allowing an alternative path for the gradient to flow through. 

#THE NETWORK STRUCTURE OF RESUNET.
#Unit level Conv layer Filter Stride Output size
#Input 224 × 224 × 3

#Unit level         Conv layer Filter Stride Output size

#Encoding
# Level 1           Conv 1 3 × 3/64 1 224 × 224 × 64
#                   Conv 2 3 × 3/64 1 224 × 224 × 64
# Level 2           Conv 3 3 × 3/128 2 112 × 112 × 128
#                   Conv 4 3 × 3/128 1 112 × 112 × 128
# Level 3           Conv 5 3 × 3/256 2 56 × 56 × 256
#                   Conv 6 3 × 3/256 1 56 × 56 × 256
# Bridge Level 4    Conv 7 3 × 3/512 2 28 × 28 × 512
#                   Conv 8 3 × 3/512 1 28 × 28 × 512

# Decoding
# Level 5           Conv 9 3 × 3/256 1 56 × 56 × 256
#                   Conv 10 3 × 3/256 1 56 × 56 × 256
# Level 6           Conv 11 3 × 3/128 1 112 × 112 × 128
#                   Conv 12 3 × 3/128 1 112 × 112 × 128
# Level 7           Conv 13 3 × 3/64 1 224 × 224 × 64
#                   Conv 14 3 × 3/64 1 224 × 224 × 64
# Output            Conv 15 1 × 1 1 224 × 224 × 1




import cv2
import numpy as np
import pandas as pd
#import os
import glob
from tqdm import tqdm



Height = 128
Width = 128



path = "E:/Data-sets/Melanoma/"
imagespath = path + "ISIC2018_Task1-2_Training_Input/*.jpg"
maskPath = path + "ISIC2018_Task1_Training_GroundTruth/*.png"

#print(imagespath)

print("Images in folder :")
listOfimages = glob.glob(imagespath)
listOfMaskImages  = glob.glob(maskPath)
#print(images)
print(len(listOfimages))
print(len(listOfMaskImages))


#print(images[0])

#lStep 1 - load one image and one mask
img = cv2.imread(listOfimages[0], cv2.IMREAD_COLOR)
print(img.shape)
#print(img.dtype)
img = cv2.resize(img, (Width, Height))
mask = cv2.imread(listOfMaskImages[0], cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask, (Width, Height))


# step2 - create sample of augmentation :
import imgaug as ia
import imgaug.augmenters as iaa

hflip= iaa.Fliplr(p=1.0)
hflipImg = hflip.augment_image(img)

vflip= iaa.Flipud(p=1.0) 
vflipImg= vflip.augment_image(img)

rot1 = iaa.Affine(rotate=(-50,20))
rotImg = rot1.augment_image(img)

cv2.imshow("img",img)
cv2.imshow("hflipimg",hflipImg)
cv2.imshow("vflipimg",vflipImg)
cv2.imshow("rotImg",rotImg)


# look at the mask
mask16 = cv2.resize(mask, (16, 16))
print(mask16) 

# 0 is the background and 255 is the Object (white object)
#We prefer than the values will be 0 and 1
mask16[mask16 > 0] = 1 

print("========================================================================================")
print(mask16) 

cv2.imshow("mask",mask)

cv2.waitKey(0)



allImages = []
maskImages = []



print(len(listOfimages))
print(len(maskImages))
# load the train images and masks
print("Start loading the train images and masks ... and augment each image X3 types ..............")
for imgFile,imgMask in tqdm(zip(listOfimages,listOfMaskImages) ,  total=len(listOfimages) ) :
    
    # create the NumpyData for images
    img = cv2.imread(imgFile, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (Width, Height))
    img = img / 255.0
    img = img.astype(np.float32)
    allImages.append(img)

    mask = cv2.imread(imgMask, cv2.IMREAD_GRAYSCALE) # gray scale image
    mask = cv2.resize(mask, (Width, Height))
    mask[mask > 0] = 1 
    maskImages.append(mask)

    #DataAugmentaion
    hflip= iaa.Fliplr(p=1.0)
    hflipImg = hflip.augment_image(img)
    hflipMask = hflip.augment_image(mask)
    allImages.append(hflipImg)
    maskImages.append(hflipMask)


    vflip= iaa.Flipud(p=1.0) 
    vflipImg= vflip.augment_image(img)
    vflipMask = vflip.augment_image(mask)
    allImages.append(vflipImg)
    maskImages.append(vflipMask)

    rot1 = iaa.Affine(rotate=(-50,20))
    rotImg = rot1.augment_image(img)
    rotMask = rot1.augment_image(mask)
    allImages.append(rotImg)
    maskImages.append(rotMask)


allImageNP = np.array(allImages)
maskImagesNP = np.array(maskImages)
maskImagesNP = maskImagesNP.astype(int) # convert to integer




print("shapes of train images and masks :")
print(allImageNP.shape)
print(maskImagesNP.shape)
print(maskImagesNP.dtype)


# load the validate images and masks
# validate

imagespath = path + "ISIC2018_Task1-2_Validation_Input/*.jpg"
maskPath = path + "ISIC2018_Task2_Validation_GroundTruth/*.png"
allValidateImages = []
maskValidateImages = []


print("Images in folder :")
listOfimages = glob.glob(imagespath)
listOfMaskImages  = glob.glob(maskPath)


print("Start loading the validate images and masks  + augmentation ................")
for imgFile,imgMask in tqdm(zip(listOfimages,listOfMaskImages) ,  total=len(listOfimages) ) :
    
    # create the NumpyData for images
    img = cv2.imread(imgFile, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (Width, Height))
    img = img / 255.0
    img = img.astype(np.float32)
    allValidateImages.append(img)

    mask = cv2.imread(imgMask, cv2.IMREAD_GRAYSCALE) # gray scale image
    mask = cv2.resize(mask, (Width, Height))
    mask[mask > 0] = 1 
    maskValidateImages.append(mask)

    #DataAugmentaion
    hflip= iaa.Fliplr(p=1.0)
    hflipImg = hflip.augment_image(img)
    hflipMask = hflip.augment_image(mask)
    allValidateImages.append(hflipImg)
    maskValidateImages.append(hflipMask)


    vflip= iaa.Flipud(p=1.0) 
    vflipImg= vflip.augment_image(img)
    vflipMask = vflip.augment_image(mask)
    allValidateImages.append(vflipImg)
    maskValidateImages.append(vflipMask)

    rot1 = iaa.Affine(rotate=(-50,20))
    rotImg = rot1.augment_image(img)
    rotMask = rot1.augment_image(mask)
    allValidateImages.append(rotImg)
    maskValidateImages.append(rotMask)


allValidateImageNP = np.array(allValidateImages)
maskValidateImages = np.array(maskValidateImages)
maskValidateImages = maskValidateImages.astype(int) # convert to integer

print("shapes of validate images and masks :")
print(allValidateImageNP.shape)
print(maskValidateImages.shape)
print(maskValidateImages.dtype)


# save the Data :

print("Save the Train Data :")
np.save("e:/temp/Unet-Train-Melanoa-Images.npy", allImageNP)
np.save("e:/temp/Unet-Train-Melanoa-Masks.npy", maskImagesNP)

print("Save the Test Data : ")
np.save("e:/temp/Unet-Test-Melanoa-Images.npy", allValidateImageNP)
np.save("e:/temp/Unet-Test-Melanoa-Masks.npy", maskValidateImages)

print("Finish save the Data ..........................")