#load and prepare the data :

# Dataset :
# https://www.kaggle.com/datasets/slothkong/10-monkey-species

import glob
import os 

print ("Start prepare the data :")
print ("========================")

trainMyImagesFolder = "C:/Python-cannot-upload-to-GitHub/10-Monkey-Species/training/training"
testMyImagesFolder = "C:/Python-cannot-upload-to-GitHub/10-Monkey-Species/validation/validation"

# check the folder

def checkMyDir(dir):
    folders = len(glob.glob(dir + '/*'))
    image_files = len(glob.glob(dir + '/*/*.jpg'))
    print ("--->>> The Data folder : {} contains {} folders and {} images.".format(dir,folders,image_files))


# print data infomation about the folders
print(checkMyDir(trainMyImagesFolder))
print(checkMyDir(testMyImagesFolder))

# lets read the monkey_lables.txt file
columns = ["Label","Latin Name","Common Name","Train Images","Validation Images"]

import pandas as pd

df = pd.read_csv("C:/Python-cannot-upload-to-GitHub/10-Monkey-Species/monkey_labels.txt",names=columns,skiprows=1)

df['Label'] = df['Label'].str.strip()
df['Latin Name'] = df['Latin Name'].replace("\t","")
df['Latin Name'] = df['Latin Name'].str.strip()
df['Common Name'] = df['Common Name'].str.strip()

df= df.set_index("Label")

print(df)

#lets create a short pandas list of the lables and the common monkey name:

monkeyDic = df["Common Name"]
print(monkeyDic)

print(monkeyDic['n0'])


# lets show some images randomly
# each columns will hold 6 images

import matplotlib.pyplot as plt 
import random

def displayDirectory(dir):
    folderList = os.listdir(dir)
    folderList.sort()

    numOfClasses = len(folderList)
    columnForDisplay = 6

    fig , ax = plt.subplots(numOfClasses, columnForDisplay, figsize=(3*columnForDisplay , 3*numOfClasses)) # space for the images

    for countRow , folderClassItem in enumerate(folderList):
        path = os.path.join(dir,folderClassItem)
        subDirList = os.listdir(path)
        #print(subDirList)

        #now , lets load 6 random images in each category

        for i in range(columnForDisplay): # 0 to 5
            randomImageFile = random.choice(subDirList)
            imageFilePath = os.path.join(path,randomImageFile)
            #print(imageFilePath)
            img = plt.imread(imageFilePath)
            monkeyLabel = monkeyDic[folderClassItem]
            monkeyLabel = monkeyLabel[:10] # get only first 10 characters

            ax[countRow,i].set_title(monkeyLabel)
            ax[countRow,i].imshow(img)
            ax[countRow,i].axis('off')

    plt.show()


displayDirectory(trainMyImagesFolder)


