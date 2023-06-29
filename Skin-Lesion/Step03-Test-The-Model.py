import tensorflow as tf 

import numpy as np
import cv2

model_file = "C:/Data-Sets/Skin-Lesion/best.h5"
model = tf.keras.models.load_model(model_file)
print(model.summary())

input_shape = (64,64)
batch= 32
categories = ['MEL', 'NV', 'BCC']

def prepareImage(img):
    resized = cv2.resize(img, input_shape, interpolation=cv2.INTER_AREA)
    imgResult = np.expand_dims(resized, axis = 0)
    imgResult = imgResult / 255.
    return imgResult


# load the google image
imgPath = "TensorFlowProjects\Skin-Lesion\Basal-cell-carcinoma.jpg"
img = cv2.imread(imgPath)

imgForModel = prepareImage(img)

# run the prediction
resultArray = model.predict(imgForModel, batch_size=batch, verbose=1)
answers = np.argmax(resultArray, axis=1)

print(answers)

text = categories[answers[0]]
print("Predicted answer is : "+ text)

font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img,text,(0,20),font,0.5,(209,19,77),2)

cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
