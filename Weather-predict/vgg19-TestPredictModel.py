import tensorflow as tf
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array

import numpy as np

categories = ["cloudy" , "foggy" , "rainy" , "shine" , "sunrise"]

model = tf.keras.models.load_model("C:/Python-cannot-upload-to-GitHub/Weather/weather-data/MyVgg19.h5")

def prepareImage(pathForImage):
    image = load_img(pathForImage , target_size=(150,150))
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult , axis = 0)
    imgResult = imgResult / 255. 
    return imgResult

testImagePath = "C:/Python-cannot-upload-to-GitHub/Weather/Test/rain_4.jpg"

# prepare the image using our function
imgForModel = prepareImage(testImagePath)

resultArray = model.predict(imgForModel , verbose=1)
#print (resultArray)

# the highest value is the predicted value and wee need the index in the resultArray
answer = np.argmax(resultArray, axis=1)
print (answer)

index = answer[0]

print ("this image is : "+ categories[ index ])

