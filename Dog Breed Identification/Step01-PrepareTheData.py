# dataset : https://www.kaggle.com/competitions/dog-breed-identification/data

import numpy as np
import cv2

IMAGE_SIZE = (331,331)
IMAGE_FULL_SIZE = (331,331,3)

trainMyImageFolder = "E:/Data-sets/Dog Breed Identification/train"

# load the csv file
import pandas as pd

df = pd.read_csv('E:/Data-sets/Dog Breed Identification/labels.csv')
print("head of lables :")
print("================")

print(df.head())
print(df.describe())

print("Group by labels : ")
grouplables = df.groupby("breed")["id"].count()
print(grouplables.head(10))

# display one image

imgPath = "E:/Data-sets/Dog Breed Identification/train/00a366d4b4a9bbb6c8a63126697b7656.jpg"
img = cv2.imread(imgPath)
#cv2.imshow("img",img)
#cv2.waitKey(0)


# prepare all the images and lables as Numpy arrays

allImages = []
allLabes = []
import os

for ix , (image_name , breed) in enumerate(df[['id' , 'breed']].values):
    img_dir = os.path.join(trainMyImageFolder, image_name + '.jpg')
    print(img_dir)

    img = cv2.imread(img_dir)
    resized = cv2.resize(img,IMAGE_SIZE, interpolation= cv2.INTER_AREA)
    allImages.append(resized)
    allLabes.append(breed)

print(len(allImages))
print(len(allLabes))

print("save the data :")
np.save("e:/temp/allDogsImages.npy",allImages)
np.save("e:/temp/allDogsLables.npy",allLabes)

print("finish save the data ....")

