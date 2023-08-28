import numpy as np
import tensorflow as tf
import cv2


best_model_file="e:/temp/Animals-Unet.h5"
model = tf.keras.models.load_model(best_model_file)
print(model.summary())

Height = 128
Width= 128
NumOfCategories = 3

allTestImagesNP = np.load("e:/temp/Unet-Animals-test-images.npy")
maskTestImagesNP = np.load("e:/temp/Unet-Animals-test-mask.npy")                     

maskTestImagesNP = maskTestImagesNP -1 

from keras.utils import np_utils
maskImagesForModel = np_utils.to_categorical(allTestImagesNP,num_classes=NumOfCategories)
maskImagesForModel = maskImagesForModel.astype(int)

print("Shapes : ")
print(allTestImagesNP.shape)
print(maskTestImagesNP.shape)

# show one image
img = allTestImagesNP[4] # image no. 4
imgForModel = np.expand_dims(img, axis=0)

p = model.predict(imgForModel)
print(p)

resultMask = p[0]
print(resultMask.shape) # we can see it is an image for 3 types of mask

# lets get the index of the high values of the last dimention
# if we have 128X128X3 (= 3 categories , not colors) , we will get the 128X128 values with the high score

resultMask = np.argmax(resultMask, axis= -1)
print ("Result after aregmax axis -1 :")
print(resultMask.shape) 

# now lets add anoher dimention 
resultMask = np.expand_dims(resultMask , axis=-1)
print("result after expand dims -1")
print(resultMask.shape) 

resultMask = resultMask * (255 / NumOfCategories)
resultMask = resultMask.astype(np.uint8)

# lets show the result .
# first with reduced size : 16X16

x = cv2.resize(resultMask, (16,16), interpolation=cv2.INTER_NEAREST)
print(x) # we can the mask in a small shape 


#lets display all together :

predictedMakImg = np.concatenate([resultMask, resultMask, resultMask], axis=2)

cv2.imshow("original image ", img)
cv2.imshow("Predicted mask ", predictedMakImg)

gray = predictedMakImg.copy()
gray = cv2.cvtColor(gray , cv2.COLOR_BGR2GRAY)
print("Gray Shape", gray.shape)

unique_vals = np.unique(gray)
print("Unique : ", unique_vals.shape)

#convert the object and the border to white color
gray[gray == 170] = 255
gray[gray == 0] = 255

# convert all the rest to black
gray[gray == 85] = 0

cv2.imshow("Gray", gray)

# apply the mask to the image and extract the object
masked_img = cv2.bitwise_and(img, img, mask=gray)
masked_img = cv2.resize(masked_img, (256,256))
cv2.imshow("masked_img", masked_img)

cv2.waitKey(0)


