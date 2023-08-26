import numpy as np
import tensorflow as tf
import cv2

#load the model
best_model_file = "e:/temp/Human-Unet.h5"

model = tf.keras.models.load_model(best_model_file)
print(model.summary())

Height = 256
Width = 256

# show one image 

imgPath = "TensorFlowProjects/Unet-Projects/Human Image Segmentation/One-Human.jpg"

img = cv2.imread(imgPath , cv2.IMREAD_COLOR)

img2 = cv2.resize(img, (Width, Height))
img2 = img2 / 255.0
imgForModel = np.expand_dims(img2, axis=0)

p = model.predict(imgForModel)
resultMask = p[0]

print(resultMask.shape)

# since it is a binary classification , any value above 0.5 means predict 1
# and any value under 0.5 is predicted to 0

# 0> black
# 1> white (mask)

resultMask[resultMask <= 0.5] = 0
resultMask[resultMask > 0.5] = 255

scale_precent = 25

width = int(img.shape[1] * scale_precent / 100)
height = int(img.shape[0] * scale_precent / 100)
dim = (width , height)

img = cv2.resize(img , dim , interpolation = cv2.INTER_AREA)
mask = cv2.resize(resultMask, dim , interpolation = cv2.INTER_AREA)

cv2.imshow("image ", img)
cv2.imshow("Mask", mask )

cv2.waitKey(0)

cv2.imwrite("e:/temp/testMask.png", mask)
