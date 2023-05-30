import tensorflow as tf
from keras.utils import img_to_array , load_img
import numpy as np
import cv2
import os

model_file = "e:/temp/emotion.h5"
model = tf.keras.models.load_model(model_file)
print(model.summary())

batchSize = 32

#get categories
print("Categories :")
trainPath = "C:/Data-Sets/Emotion-Faces/train"
categories = os.listdir(trainPath)
categories.sort()
print(categories)
numOfClasses = len(categories)
print(numOfClasses)

# find the face inside an image
# download the haarcascade from my Github

def findFace(pathForImage):
    image = cv2.imread(pathForImage)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    haarCascadeFile="e:/temp/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haarCascadeFile)
    faces = face_cascade.detectMultiScale(gray)

    for (x,y,w,h) in faces :
        #cv2.rectangle(gray, (x,y),(x+w, y+h), (255,0,0), 2)
        roiGray = gray[y:y+h , x:x+w]
        

    return roiGray
    #cv2.imshow("img",gray)
    #cv2.waitKey(0)



def prepareImageForModel(faceImage):
    resized = cv2.resize(faceImage, (48,48), interpolation=cv2.INTER_AREA)
    imgResult = np.expand_dims(resized, axis=0)
    imgResult = imgResult / 255.0
    return imgResult




# test Image
testImagePath = "TensorFlowProjects\Emotion-Detection\suprised.jpg"
#testImagePath = "TensorFlowProjects\Emotion-Detection\Happy.jpg"

faceGrayImage = findFace(testImagePath)

imgForModel = prepareImageForModel(faceGrayImage)


# run the prediction
resultArray = model.predict(imgForModel, verbose=1)
answers = np.argmax(resultArray, axis=1)

print(answers[0])

text = categories[answers[0]]

print("Predicted : " + text)

# show the image with the text
img = cv2.imread(testImagePath)
font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img, text, (0,20), font, 0.5, (209,19,77), 2)
cv2.imshow("img",img)

cv2.waitKey(0)


cv2.destroyAllWindows()

#6
#['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

#cv2.imshow("img",faceGrayImage)
#cv2.waitKey(0)
