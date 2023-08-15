import numpy as np
import tensorflow as tf
import cv2

best_model_file = "e:/temp/PolypSegment.h5"
model= tf.keras.models.load_model(best_model_file)
#print(model.summary())


# load test data

X_test = np.load("e:/temp/Unet-Polip-images-X_test.npy")
y_test = np.load("e:/temp/Unet-Polip-images-y_test.npy")

# show one of the images

img = X_test[4]

#cv2.imshow("img",img)
#cv2.waitKey(0)

# prepare for prediction
imgForModel = np.expand_dims(img, axis=0)

print(img.shape)
print(imgForModel.shape )

p = model.predict(imgForModel)
result = p[0]

print(result.shape)
#print(result)

# since it is a binary classification , so the value above 0.5 means predict to 1 and under 0.5 predict to 0
# so under 0.5 is black and above 0.5 is white

result[result <=0.5] =0
result[result > 0.5] = 255

cv2.imshow("image", img)
cv2.imshow("result mask ", result)

