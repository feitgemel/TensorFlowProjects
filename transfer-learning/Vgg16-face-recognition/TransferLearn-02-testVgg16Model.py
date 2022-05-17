import tensorflow as tf
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import load_img , img_to_array
import numpy as np

categories = [ 'Eran', 'Hana' , 'Lilach' ]

# load the model
model = tf.keras.models.load_model('C:/Python-cannot-upload-to-GitHub/MyImages/myFaceModel.h5')


# prepare the image 

def prepareImage(pathForImage) :
    image = load_img(pathForImage , target_size=(224,224))
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult , axis=0 )
    imgResult = imgResult / 255.
    return imgResult


testImage = "C:/Python-cannot-upload-to-GitHub/MyImages/Test/lilach.jpg"

imgForModel = prepareImage(testImage)
resultArray = model.predict(imgForModel,verbose=1)
print(resultArray)

answer = np.argmax(resultArray, axis=1)
print(answer)

index = answer[0]
text = "This is : "+ categories[index]
print (text)

import cv2
img = cv2.imread(testImage)

scale = 30
width = int(img.shape[1] * scale / 100)
height = int(img.shape[0] * scale / 100)
dim = (width, height)

resized = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)

cv2.putText(resized, text , (10,100), cv2.FONT_HERSHEY_COMPLEX , 1.6 , (255,0,0), 3, cv2.LINE_AA )
cv2.imshow('image', resized)
cv2.waitKey(0)