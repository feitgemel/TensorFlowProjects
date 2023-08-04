import tensorflow as tf
import os
import numpy as np
import cv2

modelFile = "e:/temp/dogs.h5"
model = tf.keras.models.load_model(modelFile)

#print(model.summary() )

inputShape = (331,331)

allLabes = np.load("e:/temp/allDogsLables.npy")
categories = np.unique(allLabes)

# prepare Image

def prepareImage(img):
    resized = cv2.resize(img, inputShape, interpolation=cv2.INTER_AREA)
    imgResult = np.expand_dims(resized, axis=0)
    imgResult = imgResult / 255.
    return imgResult

testImagePath = "TensorFlowProjects/Dog Breed Identification - Pre-trained - Works !!/Irish_Water_Spaniel1.png"
#testImagePath = "TensorFlowProjects/Dog Breed Identification - Pre-trained - Works !!/soft-coated_wheaten_terrier.jpg"
#load image
img = cv2.imread(testImagePath)
imageForModel = prepareImage(img)

# predicition

resultArray = model.predict(imageForModel, verbose=1)
answers = np.argmax(resultArray, axis = 1)

print(answers)

text = categories[answers[0]]

print(text)

font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img, text , (0,20), font , 0.5, (209,19,77), 2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.imwrite("e:/temp/img1.jpg",img)
cv2.destroyAllWindows()