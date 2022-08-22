from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint

import numpy as np
import time
import matplotlib.pyplot as plt 
import random
import glob


# Parameters

IMG=200
IMG_SIZE = [IMG,IMG]

NumOfClasses = 10
BatchSize = 32


#best_model_file = "C:/Python-cannot-upload-to-GitHub/10-Monkey-Species/myCnnMonkeyModel.h5"
#best_model_file = "C:/Python-cannot-upload-to-GitHub/10-Monkey-Species/myCnnMonkeyModelHyperParam.h5"
#best_model_file = "C:/Python-cannot-upload-to-GitHub/10-Monkey-Species/myCnnMonkeyModelHyperBandParam.h5"
best_model_file = "C:/Python-cannot-upload-to-GitHub/10-Monkey-Species/myTransferLearningMonkeyModel.h5"
model = load_model(best_model_file)


testMyImagesFolder = "C:/Python-cannot-upload-to-GitHub/10-Monkey-Species/validation/validation"
test_datageb = ImageDataGenerator(rescale = 1. /255)
test_set = test_datageb.flow_from_directory(testMyImagesFolder,
                                                shuffle=False, # dont forget
                                                target_size = IMG_SIZE,
                                                batch_size = BatchSize,
                                                class_mode = 'categorical')


predictions = model.predict(test_set)
#print(type(predictions))
#print(predictions)
predictionsResults = np.argmax(predictions, axis=1)
print(predictionsResults)

# lets copy the code from the first tutorial 
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

# lets see the images randomly and compare the real label with the predicted label

def compareResults():
    image_files = glob.glob(testMyImagesFolder + '/*/*.jpg')
    nrows = 5
    ncols = 6
    picnum = nrows * ncols

    fig , ax = plt.subplots(nrows , ncols , figsize=(3*ncols , 3*nrows))
    correct = 0

    for i in range(picnum) :
        x = random.choice(image_files)
        xi = image_files.index(x) # get the position of the random image
        img1 = plt.imread(x)

        pred1 = monkeyDic[predictionsResults[xi]]
        pred1 = pred1[:7]

        real1 = monkeyDic[test_set.classes[xi]]
        real1 = real1[:7]

        if (pred1 == real1 ):
            correct = correct + 1

        name = 'predicted : {} \nreal: {}'.format(pred1,real1)
        plt.imshow(img1)
        plt.title(name)

        sp = plt.subplot(nrows,ncols, i+1 )
        sp.axis('off') # no grid

    print(" ======================================================= ")
    print("Total : {} correct {} : ".format(picnum , correct))

    plt.show()



# run the function
compareResults()
