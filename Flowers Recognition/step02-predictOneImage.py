import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import cv2
import os
import glob


#lets download some new images from google , show the image , choose one using "Enter" key and predict it
# then , we will show the image with its predictions as one image

# this folder of images will be in my Github 

# flower categories :
# important - the list should be sorted 
flower_categories = ['daisy', 'dandelion' , 'rose', 'sunflower' , 'tulip']


#load the saved model :
model = tf.keras.models.load_model('C:/Python-cannot-upload-to-GitHub/flowers/flowers.h5')

img_dir = "C:/GitHub/TensorFlowProjects/Flowers Recognition/FromGoogle"
data_path = os.path.join(img_dir,'*')
files = glob.glob(data_path)

num = 0 

for f1 in files:
    num = num + 1
    img = cv2.imread(f1)
    cv2.imshow('img', img)
    key = cv2.waitKey(0)

    # if <Enter> pressed than break the loop -> this is the chosen image
    if key == 13:
        break

# the chosen image
print(f1)

test_image = image.load_img(f1, target_size=(224,224))

#convert the image into array
test_image = image.img_to_array(test_image)

#expand the array with another dimenation
test_image = np.expand_dims(test_image, axis=0)

# predict the category of an image 
result = model.predict(test_image)

print(result[0])

#The position of the 1 value is the prediciton of the category 

indPositionMax = np.argmax(result[0])

print('The position is : ',indPositionMax )

flower_predict = flower_categories[indPositionMax]
text = "Prediction : " + flower_predict

imgFinalResult = cv2.imread(f1)
font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(imgFinalResult, text , (0,100), font, 2, (255,0,0), 3)
cv2.imshow('img', imgFinalResult )
cv2.waitKey(0)

cv2.imwrite('C:/Python-cannot-upload-to-GitHub/flowers/result.jpg',imgFinalResult)