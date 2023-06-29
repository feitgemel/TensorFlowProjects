# https://www.kaggle.com/datasets/andrewmvd/isic-2019?select=ISIC_2019_Training_GroundTruth.csv

import numpy as np
import pandas as pd
import cv2

csv_file = "C:/Data-Sets/Skin-Lesion/ISIC_2019_Training_GroundTruth.csv"
df = pd.read_csv(csv_file , header=0)
print(df.head(20))

print("Rows and Columns :")
print(df.shape)

pathToImages = "C:/Data-Sets/Skin-Lesion/ISIC_2019_Training_Input/ISIC_2019_Training_Input"

# get the column names
categories = list(df.columns)
print(categories)

categories.remove('image')
print("categories : "+ str(categories))


# arrays for images and lables

allImages64=[]
gray64=[]
allLables=[]
input_shape64 = (64,64)


for index in df.index:

    name = df['image'][index]
    fileName = pathToImages+"/"+df['image'][index] + ".jpg"

    #print(fileName)

    #load images
    img = cv2.imread(fileName)
    #cv2.imshow("img",img)
    #cv2.waitKey(0)
    if img is None:
        print("Failed to load image "+ fileName)
    else :

        # get the lables 
        values=[]
        values.append(df['MEL'][index])
        values.append(df['NV'][index])
        values.append(df['BCC'][index])
        values.append(df['AK'][index])
        values.append(df['BKL'][index])
        values.append(df['DF'][index])
        values.append(df['VASC'][index])
        values.append(df['SCC'][index])
        values.append(df['UNK'][index])

        # the position in the array -> each category is integer between 0 to 8

        ind = values.index(1.0)

        # let's start with only 3 categories :

        if ind ==0 or ind ==1 or ind == 2 :
            allLables.append(ind)

            resized64 = cv2.resize(img, input_shape64, interpolation=cv2.INTER_AREA)
            allImages64.append(resized64)

            grayImage64 = cv2.cvtColor(resized64, cv2.COLOR_BGR2GRAY)
            gray64.append(grayImage64)


            print("Filename "+name + " is : "+str(ind)+ " shape :" + str(resized64.shape))


#print(len(allImages64))
#print(len(allLables))


allImages64 = np.array(allImages64)
allImagesGray64 = np.array(gray64)

allLables = np.array(allLables)


print(len(allImages64))
print(len(allLables))

print("Save the Data :")
np.save("C:/Data-Sets/Skin-Lesion/allImages64.npy",allImages64)
np.save("C:/Data-Sets/Skin-Lesion/allImagesGray64.npy",allImagesGray64)
np.save("C:/Data-Sets/Skin-Lesion/allLables.npy",allLables)

print("Finish save the Data")











