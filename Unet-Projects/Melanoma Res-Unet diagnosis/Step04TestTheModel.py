import numpy as np
import tensorflow as tf
import cv2

#load the model

best_model_file = "e:/temp/MelanomaResUnet.h5"
model = tf.keras.models.load_model(best_model_file)
print(model.summary())

Height=128
Width = 128

# show one of the test images

img = cv2.imread("E:/Data-sets/Melanoma/ISIC2018_Task1-2_Test_Input/ISIC_0012302.jpg",cv2.IMREAD_COLOR)


# prerpare img for the model
img2 = cv2.resize(img,(Width,Height))
img2 = img2 / 255.0
imgForModel = np.expand_dims(img2, axis=0)

p = model.predict(imgForModel)
resultMask = p[0]
print(resultMask.shape)


# since it is a binary classification
# all values under 0.5 will be replace to 0 (black)
# all values above 0.5 will be replaced to 255 (white)

resultMask[resultMask <=0.5 ] = 0 # black
resultMask[resultMask > 0.5 ] = 255 # white




scale_precent = 25
width = int(img.shape[1] * scale_precent/100)
height = int(img.shape[0] * scale_precent/100)
dim = (width,height)
img = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)

# resize the predicted mask before display 
mask = cv2.resize(resultMask, dim , interpolation=cv2.INTER_AREA)


# lets load the ground truth
trueMaskfile = "E:/Data-sets/Melanoma/ISIC2018_Task1_Test_GroundTruth/ISIC_0012302_segmentation.png"
trueMask = cv2.imread(trueMaskfile, cv2.IMREAD_COLOR)
trueMask = cv2.resize(trueMask,dim,interpolation=cv2.INTER_AREA)

cv2.imshow("original image", img)
cv2.imshow("predicted mask ", mask)
cv2.imshow("trueMask mask ", trueMask)

cv2.imwrite("e:/temp/predicted.jpg",mask)

cv2.waitKey(0)