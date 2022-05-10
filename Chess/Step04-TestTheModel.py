from keras.models import load_model

imgWidth = 256
imgHeight = 256

# the names of the classes should be sorted 


classes = ["Bishop","King","Knight", "Pawn","Queen","Rook"]

import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img , img_to_array
import cv2

#lets load the model 

model = load_model("C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/chess_best_model.h5")

#print(model.summary() )


# lets build a function for preparing an image for model

def prepareImage(pathToImage) :
    image = load_img(pathToImage , target_size=(imgHeight, imgWidth))
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult , axis=0 )
    imgResult = imgResult / 255.
    return imgResult


#testImagePath = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/Test/knight.jpeg"
testImagePath = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/Test/rook.jpg"



# run the function
imageForModel = prepareImage(testImagePath)

print(imageForModel.shape)

#predict the image
resultArray = model.predict(imageForModel , batch_size=32 , verbose=1)
answer = np.argmax(resultArray , axis=1 )

print(answer[0])

text = classes[answer[0]]
print ('Predicted : '+ text)

#lets show the image with the predicted text 

img = cv2.imread (testImagePath)
font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img , text , (0,100) , font , 2 , (209,19,77 ) , 3 )
cv2.imshow('img', img)
cv2.waitKey(0)

