
#google models :
#https://tfhub.dev/google/collections/landmarks/1
#inside you can choose Continent (Europe , Africa , etc .....)




import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
import tensorflow_hub as hub



# copy the url from the web page : https://tfhub.dev/google/collections/landmarks/1
modelUrl = "https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_europe_V1/1"

#download the lables file from the same link 
# we have about 100,000 classes 
labels = "C:/Python-Code/ObjectDetection/Landmark-classifier/landmarks_classifier_europe_V1_label_map.csv"

# the model need image in the size of 321,321
imageShape = (321,321,3)

classifier = tf.keras.Sequential( [ hub.KerasLayer(modelUrl , input_shape=imageShape , output_key="predictions:logits"  ) ]  )


# build a key and value for the lables
df = pd.read_csv(labels)

print(df)
print(zip(df.id, df.name))
print(dict(zip(df.id, df.name))) # create a dictionaty 

labelsDict = dict(zip(df.id, df.name))

#test an image
#=============

#image1 = PIL.Image.open("C:/Python-Code/ObjectDetection/Landmark-classifier/EifelTestImage.jpg")
#image1 = PIL.Image.open("C:/Python-Code/ObjectDetection/Landmark-classifier/tower.jpg")
image1 = PIL.Image.open("C:/Python-Code/ObjectDetection/Landmark-classifier/brandenburger-tor.jpg")

image1 = image1.resize((321,321))
print(image1.size)

# convert to Numpy Araay
image1 = np.array(image1)
image1 = image1 / 255.0

print(image1.shape)

# build array of images 
image1 = image1[np.newaxis]
print(image1.shape)

result = classifier.predict(image1)
finalResult = np.argmax(result) # get the highest score index 
print ("The preidiction is : " + labelsDict[finalResult] )


