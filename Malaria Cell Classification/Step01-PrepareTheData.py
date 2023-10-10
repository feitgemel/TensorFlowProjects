# Kaggle - Search for "Malaria cell images dataset"
# or use this link : https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

import numpy as np
import pandas as pd
import cv2
import glob

allImages = []
allLables = []

input_shape = (124, 124)

ParasitedPath = "E:/Data-sets/Malaria Cell Classification/cell_images/Parasitized"
UninfectedPath = "E:/Data-sets/Malaria Cell Classification/cell_images/Uninfected"

paths = [ParasitedPath, UninfectedPath]

for path in paths:
    path2 = path + "/*.png"
    for file in glob.glob(path2):
        print(file)

        #load the image
        img = cv2.imread(file)

        if img is not None:
            resized = cv2.resize(img, input_shape, interpolation = cv2.INTER_AREA)
            allImages.append(resized)

            if path ==ParasitedPath :
                allLables.append(0)
            else : # un infected
                allLables.append(1)

allImagesNP = np.array(allImages)
print(allImagesNP.shape)

allLablesNP = np.array(allLables)
print(allLablesNP.shape)


print("Save the data")
np.save("e:/temp/Malaria-images.npy",allImagesNP)
np.save("e:/temp/Malaria-lables.npy", allLablesNP)
print("Finish save the data ....")