import tensorflow as tf
import os
from keras.utils import img_to_array , load_img
import numpy as np
import cv2

# load the model
model = tf.keras.models.load_model("e:/temp/Fruits360.h5")
print(model.summary())

# load the categories :

source_folder = "E:/Data-sets/fruits-360_dataset/fruits-360/Test"
categories = os.listdir(source_folder) 
categories.sort()
print(categories)
numofClasses = len(categories)
print(numofClasses)

# load and prepare image

def prepareImage(pathForImage):
    image = load_img(pathForImage, target_size=(100,100))
    imgResult = img_to_array(image)
    #print(imgResult.shape)
    imgResult = np.expand_dims(imgResult , axis = 0)
    #print(imgResult.shape)
    imgResult = imgResult / 255.
    return imgResult

testImagePath = "TensorFlowProjects/Fruits classification/eggplant.jpg"
imageForModel = prepareImage(testImagePath)

resultArray = model.predict(imageForModel, verbose=1)
answers = np.argmax(resultArray, axis = 1)
print(answers[0])

text = categories[answers[0]]
print("Predicted image : "+ text)

# show the image with the text

img = cv2.imread(testImagePath)
font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img,text,(0,50),font,1,(209,19,77),2)
cv2.imshow('img',img)
cv2.imwrite('TensorFlowProjects/Fruits classification/predicted.png',img)
cv2.waitKey(0)

cv2.destroyAllWindows()

