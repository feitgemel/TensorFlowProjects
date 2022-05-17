from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

from PIL import Image , ImageFile
import numpy as np
import cv2

model = MobileNet(weights="imagenet", include_top=True)

print (model.summary() )

# the model predicts 1000 classes 

from tensorflow.keras.applications.mobilenet import decode_predictions

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3

# any image should be converted to this dim : 224,224,3

# function for classify image 

def classify_image (imageFile) :
    x= []

    img = Image.open(imageFile)
    img.load()
    img = img.resize( (IMAGE_WIDTH,IMAGE_HEIGHT) , Image.ANTIALIAS)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    # process the image
    x = preprocess_input(x)

    pred = model.predict(x)

    # we got 1000 results 
    # lets print the highest score 

    print(np.argmax(pred,axis=1))
    # so it is sea snake .
    

    # get the best five results 
    list = decode_predictions(pred , top=5)
    for itm in list[0]:
        print(itm)

    result = list[0][0]
    _ , classText , p = result 

    return classText


#imagePath = "transfer-learning/MobileNet/seaSnake.jpg"
imagePath = "transfer-learning/MobileNet/Hero-Volcano.jpg"


resultText = classify_image(imagePath)

print(resultText)

# show the image

img = cv2.imread(imagePath)
img = cv2.putText(img , resultText , (50,50), cv2.FONT_HERSHEY_SIMPLEX , 0.6 , (51,255,255), 1 )
cv2.imshow('img' , img)
cv2.waitKey(0)
cv2.destroyAllWindows()