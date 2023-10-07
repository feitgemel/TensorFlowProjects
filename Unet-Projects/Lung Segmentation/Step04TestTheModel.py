import numpy as np
import tensorflow as tf
import cv2

#load the saved model
best_model_file = "e:/temp/lung-Unet.h5"
model = tf.keras.models.load_model(best_model_file)
print(model.summary())

Width = 256
Height = 256

# load test image
testImagePath = "TensorFlowProjects/Unet-Projects/Lung Segmentation/Lung-test-Image-From-Google.jpeg"
img = cv2.imread(testImagePath)
img2 = cv2.resize(img, (Width, Height))
img2 = img2 / 255.0
imgForModel = np.expand_dims(img2, axis=0)

p = model.predict(imgForModel)
resultMask = p[0]

print(resultMask.shape)


# since it is a binary classification , any value under 0.5 would be replaced with  0 (Black)
# and any value above 0.5 would we replaced with 255 (white)

resultMask[resultMask <= 0.5] = 0
resultMask[resultMask > 0.5] = 255

scale_precent = 60 # for resize and display

w = int(img.shape[1] * scale_precent / 100 )
h = int(img.shape[0] * scale_precent / 100 )

dim = (w,h)

img = cv2.resize(img, dim , interpolation = cv2.INTER_AREA)
mask = cv2.resize(resultMask, dim , interpolation = cv2.INTER_AREA)

cv2.imshow("first image", img)
cv2.imshow("Predicted mask ", mask)


cv2.waitKey(0)


