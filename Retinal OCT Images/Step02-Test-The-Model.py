import tensorflow as tf
import os 
from keras.utils import img_to_array, load_img
import numpy as np 
import cv2


# load the model
model = tf.keras.models.load_model("e:/temp/retinalOCT.h5")
#print(model.summary())


# load categories
source_folder = "E:/Data-sets/Retinal OCT Images/val"
categories = os.listdir(source_folder)
categories.sort()
print(categories)

numOfClasses = len(categories)
print(numOfClasses)


def prepareImage(pathForImage):
    image = load_img(pathForImage, target_size=(224,224), color_mode='grayscale')
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult, axis=0)
    imgResult = imgResult / 255.
    return imgResult

# prediction

#testImagePath = "E:/Data-sets/Retinal OCT Images/val/CNV/CNV-6851127-1.jpeg"
testImagePath = "E:/Data-sets/Retinal OCT Images/val/DRUSEN/DRUSEN-9894035-1.jpeg"


imgForModel = prepareImage(testImagePath)

resultArray = model.predict(imgForModel, verbose=1)
answers = np.argmax(resultArray, axis=1)

print(answers)

text = categories[answers[0]]
print("Predicted : " + text)

# show the image :

img = cv2.imread(testImagePath)
font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img, text , (0,50), font , 2, (209,19,77), 2)
cv2.imshow('img', img)
cv2.waitKey(0)

cv2.destroyAllWindows()