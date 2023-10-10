import tensorflow as tf
import os
import cv2
import numpy as np 

best_model_file = "e:/temp/Malaria_binary.h5"
model = tf.keras.models.load_model(best_model_file)
print(model.summary())

input_shape = (124, 124)
categories = ["infected" , "uninfected"]

def prepareImage(img):
    resized = cv2.resize(img , input_shape, interpolation=cv2.INTER_AREA)
    imgResult = np.expand_dims(resized , axis=0)
    imgResult - imgResult / 255.
    return imgResult

# load test image
testImagePath = "TensorFlowProjects/Malaria Cell Classification/testInfected.jpg"
img = cv2.imread(testImagePath)

# prepare image for the model
imgForModel = prepareImage(img)

# run the prediction
result = model.predict(imgForModel, verbose=1)
print(result)

# binary classification
if result > 0.5 :
    result = 1
else :
    result = 0

print(result)
text = categories[result]

# show the image + prediction

img = cv2.imread(testImagePath)
font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img , text , (0,20) , font , 1, (0,255,255), 2) #yellow
cv2.imshow("img", img)
cv2.waitKey(0)




