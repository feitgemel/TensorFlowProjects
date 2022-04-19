import os
import random
import shutil

dataOrgFolder = "C:/Python-cannot-upload-to-GitHub/Weather/original-dataset/"
dataBaseFolder = "C:/Python-cannot-upload-to-GitHub/Weather/dataset"

dataDirList = os.listdir(dataOrgFolder)
print(dataDirList)

splitSize = .85

# build files array

def split_data (SOURCE , TRAINING , VALIDATION , SPLIT_SIZE):
    files = []

    for filename in os.listdir(SOURCE) :
        file = SOURCE + filename
        print(file)
        if os.path.getsize(file) > 0 :
            files.append(filename)
        else:
            print(filename + " has 0 length , will not copy this file !!")

    # print number of files :
    print(len(files))


    trainLength = int(len(files) * SPLIT_SIZE )
    validLength = int( len(files) - trainLength )

    suffleDataSet = random.sample(files, len(files))

    trainingSet = suffleDataSet[0:trainLength]
    validSet = suffleDataSet[trainLength:]

    # copy the train images 
    for filename in trainingSet:
        f = SOURCE + filename
        dest = TRAINING + filename
        shutil.copy(f, dest) 
    
    # copy the valid images 
    for filename in validSet:
        f = SOURCE + filename
        dest = VALIDATION + filename
        shutil.copy(f, dest) 



cloudySourceFolder = "C:/Python-cannot-upload-to-GitHub/Weather/original-dataset/cloudy/"
cloudyTrainFolder = "C:/Python-cannot-upload-to-GitHub/Weather/dataset/Train/cloudy/"
cloudyValidFolder = "C:/Python-cannot-upload-to-GitHub/Weather/dataset/validate/cloudy/"

foggySourceFolder = "C:/Python-cannot-upload-to-GitHub/Weather/original-dataset/foggy/"
foggyTrainFolder = "C:/Python-cannot-upload-to-GitHub/Weather/dataset/Train/foggy/"
foggyValidFolder = "C:/Python-cannot-upload-to-GitHub/Weather/dataset/validate/foggy/"

rainyySourceFolder = "C:/Python-cannot-upload-to-GitHub/Weather/original-dataset/rainy/"
rainyTrainFolder = "C:/Python-cannot-upload-to-GitHub/Weather/dataset/Train/rainy/"
rainyValidFolder = "C:/Python-cannot-upload-to-GitHub/Weather/dataset/validate/rainy/"

shineSourceFolder = "C:/Python-cannot-upload-to-GitHub/Weather/original-dataset/shine/"
shineTrainFolder = "C:/Python-cannot-upload-to-GitHub/Weather/dataset/Train/shine/"
shineValidFolder = "C:/Python-cannot-upload-to-GitHub/Weather/dataset/validate/shine/"

sunriseSourceFolder = "C:/Python-cannot-upload-to-GitHub/Weather/original-dataset/sunrise/"
sunriseTrainFolder = "C:/Python-cannot-upload-to-GitHub/Weather/dataset/Train/sunrise/"
sunriseValidFolder = "C:/Python-cannot-upload-to-GitHub/Weather/dataset/validate/sunrise/"






split_data(cloudySourceFolder , cloudyTrainFolder , cloudyValidFolder , splitSize)
split_data(foggySourceFolder , foggyTrainFolder , foggyValidFolder , splitSize)
split_data(rainyySourceFolder , rainyTrainFolder , rainyValidFolder , splitSize)
split_data(shineSourceFolder , shineTrainFolder , shineValidFolder , splitSize)
split_data(sunriseSourceFolder , sunriseTrainFolder , sunriseValidFolder , splitSize)




# split_data ( "C:/Python-cannot-upload-to-GitHub/Weather/original-dataset/cloudy/", # dont forget the last " / "
#     "C:/Python-cannot-upload-to-GitHub/Weather/dataset/Train/cloudy/",
#     "C:/Python-cannot-upload-to-GitHub/Weather/dataset/validate/cloudy/",
#     splitSize)

