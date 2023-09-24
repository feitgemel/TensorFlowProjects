import numpy as np
import tensorflow as tf
import cv2

#load the model

best_model_file = "e:/temp/melanoma-Unet.h5"
model = tf.keras.models.load_model(best_model_file)
print(model.summary())

Height= 128
Width=128

imgTestPath= "E:/Data-sets/Melanoma/ISIC2018_Task1-2_Validation_Input/ISIC_0012643.jpg"
img = cv2.imread(imgTestPath)


img2 = cv2.resize(img , (Width, Height))
img2 = img2 / 255.0
imgForModel = np.expand_dims(img2, axis=0)

print(img2.shape)
print(imgForModel.shape)

p = model.predict(imgForModel)
resultMask = p[0]
print("Result mask : ")
print(resultMask.shape)


# since it is a binary classification , any value above 0.5 will be changed to 255 (White)
# and any value under 0.5 will be changed to 0 (Black)

# this will create a predicted mask !!!

resultMask[resultMask <=0.5] = 0
resultMask[resultMask > 0.5] = 255

scale_precent = 25
width = int(img.shape[1] * scale_precent / 100)
height = int(img.shape[0] * scale_precent / 100) 
dim = (width, height)

img = cv2.resize(img, dim , interpolation = cv2.INTER_AREA)
mask = cv2.resize(resultMask, dim , interpolation = cv2.INTER_AREA)

trueMaskFile = "E:/Data-sets/Melanoma/ISIC2018_Task2_Validation_GroundTruth/ISIC_0012643_attribute_pigment_network.png"
trueMask = cv2.imread(trueMaskFile,cv2.IMREAD_COLOR)
trueMask = cv2.resize(trueMask,  dim , interpolation = cv2.INTER_AREA)

# show the result
cv2.imshow("First image", img)
cv2.imshow("Predicted Mask ", mask)
cv2.imshow("Real mask ", trueMask)

cv2.imwrite("e:/temp/predicted.png",mask)
cv2.imwrite("e:/temp/img.png",img)

cv2.waitKey(0)


