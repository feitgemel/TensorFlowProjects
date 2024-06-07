import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import cv2

# dataset
#https://www.kaggle.com/puneet6060/intel-image-classification

modelUrl = "https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_europe_V1/1"

# lets download the label from the same link
labels = "C:/GitHub/TensorFlowProjects/Tensor-Hub-Landmarks/landmarks_classifier_europe_V1_label_map.csv"

imageShape = (321,321,3)
classifier = tf.keras.Sequential(
    [hub.KerasLayer(modelUrl , input_shape = imageShape , output_key="predictions:logits")])

df = pd.read_csv(labels)

print(df)

print(zip(df.id , df.name))

# create a dictinary
labelsDict = dict(  zip(df.id , df.name) )

# test an image
imagePath = "C:/GitHub/TensorFlowProjects/Tensor-Hub-Landmarks/tower.jpg"
image1 = PIL.Image.open(imagePath)
image1 = image1.resize((321,321))
print(image1.size)

#convert the image to Numpy array
image1 = np.array(image1)
image1 = image1 / 255.0

image1 = image1[np.newaxis]

# it sholud be array of images - missing layer 
print(image1.shape )

result = classifier.predict(image1)
finalResult = np.argmax(result)

text = "The predicition is : " + labelsDict[finalResult]
print (text)

img = cv2.imread(imagePath)
font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img, text , (50,100), font , 2 ,(0,255,0), 1)
cv2.imshow('img', img)
cv2.waitKey(0)

