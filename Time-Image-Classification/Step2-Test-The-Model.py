from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
import matplotlib.pyplot as plt
import numpy as np
import random
import os


# copy from other previous python script
IMG = 224
IMG_SIZE = [IMG,IMG]

numOfClasses = 144
batchSize = 64
best_model_file = "C:/Data-Sets/TIME -Image Dataset-Classification/myTimeCnn.h5"

model = load_model(best_model_file)

#print(model.summary())

# load the test data
TestMyImagesFolder = "C:/Data-Sets/TIME -Image Dataset-Classification/test"

categories = os.listdir(TestMyImagesFolder)
categories.sort()
print(categories)

testDataGen = ImageDataGenerator(rescale = 1. / 255)
testSet = testDataGen.flow_from_directory(TestMyImagesFolder,
                                        target_size = IMG_SIZE,
                                        shuffle = False,
                                        batch_size = batchSize,
                                        color_mode = "grayscale",
                                        class_mode = "categorical")

predictions = model.predict(testSet)

print(predictions.shape)

# we can see that for each of the 144- images we got 144 results (result for each category)
# we are looking for the high "Score" in each of the images

predictionsReults = np.argmax(predictions , axis = 1)
print(categories[predictionsReults[0]]) # predicted class for the first image 
print(categories[predictionsReults[100]]) # predicted class for the second image 

# lest see the images :

def compareResults():
    image_files = glob.glob(TestMyImagesFolder + '/*/*.jpg')
    nrows = 5
    ncols = 6 
    picNum = nrows * ncols

    fix , ax = plt.subplots(nrows , ncols , figsize=(3 * ncols , 3* nrows))
    correct = 0

    for i in range (picNum):
        x = random.choice(image_files)
        xi = image_files.index(x) # get the position of the random image
        img1 = plt.imread(x)

        pred1 = categories[ predictionsReults[xi]]
        pred1 = pred1[:7] # slice 7 characters from the start point of the string

        real1 = categories[testSet.classes[xi]]
        real1 = real1[:7]

        if (pred1 == real1) :
            correct = correct + 1
        
        name = "Predicted: {} \nreal:{}".format(pred1,real1)
        plt.imshow(img1)
        plt.title(name)
        sp = plt.subplot(nrows,ncols, i+ 1)
        sp.axis('off')

    print("=========================================================")
    print("Total : {} correct {}: ".format(picNum, correct))

    plt.show()
    
# run the function
compareResults()


