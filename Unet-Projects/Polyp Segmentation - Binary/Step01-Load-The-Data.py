# Dataset : https://www.kaggle.com/datasets/balraj98/cvcclinicdb

import cv2
import numpy as np
import os

height=256
width = 256

allImages = []
maskImages = []

path = "E:/Data-sets/Polyp/PNG/"
imagesPath = path + "Original"
maskPath = path + "Ground Truth"

print("Images in folder :")
images = os.listdir(imagesPath)
#print(images)
print(len(images))

# Load one image and one mask
img = cv2.imread(imagesPath+"/1.png", cv2.IMREAD_COLOR)
img = cv2.resize(img , (width,height))

mask = cv2.imread(maskPath+"/1.png", cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask , (width,height))

cv2.imshow("img", img)
cv2.imshow("mask", mask)


# lets look at the values of the mask
# resize temporary 

resizeto16 = cv2.resize(mask, (16,16))
print(resizeto16)

# We can see the the mask area is higher than 0 , and the background is 0 (Black)
# let's change the values to binary ( all the black would be 0 , and all the other will be white - 1)

resizeto16[resizeto16 <=50 ] = 0
resizeto16[resizeto16 > 50 ] = 1

print(resizeto16)

cv2.waitKey(0)

# create the Numpy arrays :

for imagefile in images :
    
    # image
    file = imagesPath + "/" + imagefile
    img = cv2.imread(file , cv2.IMREAD_COLOR)
    img = cv2.resize(img , (width, height))
    img = img / 255.0
    img = img.astype(np.float32)
    allImages.append(img)

    # mask 
    file = maskPath + "/" + imagefile
    mask = cv2.imread(file , cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask , (width, height))
    
    mask[mask <=50 ] = 0
    mask[mask > 50 ] = 1
    
    maskImages.append(mask)   

# After loop
allImagesNP = np.array(allImages)
maskImageNP = np.array(maskImages)
maskImageNP = maskImageNP.astype(int) # all the values should be interger (0 or 1)


print(allImagesNP.shape)
print(allImagesNP.dtype)

print(maskImageNP.shape)
print(maskImageNP.dtype)


# split train and test
from sklearn.model_selection import train_test_split

#90% train , 10% test
X_train , X_test , y_train , y_test = train_test_split(allImagesNP, maskImageNP , test_size=0.1, random_state=42)

# 80% train , 10% val

X_train , X_val , y_train , y_val = train_test_split(X_train, y_train, test_size=0.1 , random_state=42)


print ("X_train , X_val , y_train , x_val, X_test , y_test ------> shapes :")

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)


# save the data
print("Start save the data :")
np.save("e:/temp/Unet-Polip-images-X_train.npy",X_train)
np.save("e:/temp/Unet-Polip-images-y_train.npy",y_train)
np.save("e:/temp/Unet-Polip-images-X_val.npy",X_val)
np.save("e:/temp/Unet-Polip-images-y_val.npy",y_val)
np.save("e:/temp/Unet-Polip-images-X_test.npy",X_test)
np.save("e:/temp/Unet-Polip-images-y_test.npy",y_test)
print("Finish save the data .......")






