import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.utils import load_img
from keras.preprocessing import image 

IMAGE_SIZE = 128
BATCH_SIZE = 32
CHANNELS = 3

#load the model 

model = tf.keras.models.load_model("e:/Temp/Landscape-Model.h5")

print(model.summary())


# get the list of categories 
Train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "e:/Data-sets/Intel-images/seg_train/seg_train",
    shuffle=True,
    image_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE 
)

class_names = Train_dataset.class_names
print("class_names: " + str(class_names))


#numberOfClasses = len(class_names)
#print("number of classes : " + str(numberOfClasses))


def predictImage (model , img ) :
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 

    predictions = model.predict(img_array) 

    result = predictions[0]
    resultIndex = np.argmax(result)

    predictedClass = class_names[resultIndex]
    confidence = round (100 * np.max(result), 2)

    return predictedClass, confidence


img_path = "e:/Data-sets/Intel-images/seg_pred/seg_pred/720.jpg"

originalImage = cv2.imread(img_path)
testImage = load_img(img_path, target_size=(IMAGE_SIZE,IMAGE_SIZE))

print(type(testImage))

testImage = tf.keras.preprocessing.image.img_to_array(testImage)

print(type(testImage))
print(testImage.shape)

# run the predict function

predictedClassName , confidence = predictImage(model , testImage)

print(predictedClassName)
print(confidence)


# show to result with the image

#resize the image (larger)
scale_percent = 300
width = int(originalImage.shape[1] * scale_percent/100)
height = int(originalImage.shape[0] * scale_percent/100)    
dim = (width,height )

resized = cv2.resize(originalImage, dim , interpolation=cv2.INTER_AREA)
resized = cv2.putText(resized , predictedClassName, (10,100) , cv2.FONT_HERSHEY_COMPLEX, 1.6 , (255,255,0), 3, cv2.LINE_AA)
resized = cv2.putText(resized , str(confidence), (10,200) , cv2.FONT_HERSHEY_COMPLEX, 1.6 , (255,255,0), 3, cv2.LINE_AA)

cv2.imshow('img', resized)
cv2.waitKey(0)