import numpy as np

# load test data 

print("load test (!!!) data ")
x_test = np.load("e:/temp//5-vehicales-x_test.npy")
y_test = np.load("e:/temp//5-vehicales-y_test.npy")

print(x_test.shape)
print(y_test.shape)

# xgboost
import xgboost as xgb
model = xgb.Booster()
model.load_model("e:/temp/5-vehicles-XGboost.h5")

print("Finish load the model ")

import tensorflow as tf
model_file = "e:/temp/5-vehicales-Vgg_model.h5"
vgg_model = tf.keras.models.load_model(model_file)

import cv2

# lets get a random image out of all the test images
n = np.random.randint(0, x_test.shape[0])
print(n)

img = x_test[n]
print(img.shape)

input_img = np.expand_dims(img, axis=0) # expand dims so the input shape will be ( num of images , x, t, c )
print(input_img.shape )

# extract the features using Vgg16 model and reshape it , same as we did in the train process
input_img_feature = vgg_model.predict(input_img)

input_img_features = input_img_feature.reshape(input_img_feature.shape[0], -1)

# convert the input data to Dnatrix object
DMinput = xgb.DMatrix(input_img_features)

prediciton = model.predict(DMinput)
print(prediciton)

result = prediciton[0]
finalResult = np.argmax(result)
print(finalResult)

originalLabels = np.load("e:/temp/5-vehicales-categories.npy")
text = originalLabels[finalResult]

print("The predicition for this image is : ", text)

font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text , (0,20), font, 0.5 , (0,255,255), 1)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()








