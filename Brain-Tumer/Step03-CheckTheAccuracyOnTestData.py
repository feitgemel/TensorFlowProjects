# now , we will check our model within a new test data
# than , we will run a prediction on an image

from locale import normalize
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import keras
import cv2

TEST_DIR = "C:/Python-cannot-upload-to-GitHub/BrainTumor/test"

test_datagen = image.ImageDataGenerator( rescale= 1. / 255)
test_data = test_datagen.flow_from_directory(directory=TEST_DIR , target_size=(224,224) , batch_size=32 , class_mode='binary')

# lets print the classes :

print("test_data.class_indices: ", test_data.class_indices)

#load the saved model :

model = load_model('C:/Python-cannot-upload-to-GitHub/BrainTumor/MyBestModel.h5')

#print(model.summary() )

acc = model.evaluate(x=test_data)[1]

print(acc)



# load an image from the test folder 
imagePath = "C:/Python-cannot-upload-to-GitHub/BrainTumor/test/Healthey/Not Cancer  (1523).jpg"
#imagePath = "C:/Python-cannot-upload-to-GitHub/BrainTumor/test/Brain Tumor/Cancer (17).jpg"

img = image.load_img(imagePath,target_size=(224,224))
i = image.img_to_array(img) # convert to array
i = i / 255 # -> normalize to our model
print(i.shape)

input_arr = np.array([i]) # add another dimention 
print(input_arr.shape)


# run the prediction
predictions = model.predict(input_arr)[0][0]
print(predictions)


# since it is binary if the result is close to 0 it is Tumor , and if it close to 1 it is healthy
result = round(predictions)
if result == 0 :
    text = 'Has a brain tumor'
else :
    text = "Brain healthy"


print(text)

imgResult = cv2.imread(imagePath)
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(imgResult, text, (0,20), font, 0.8 , (255,0,0),2 )
cv2.imshow('img', imgResult)
cv2.waitKey(0)
cv2.imwrite("C:/Python-cannot-upload-to-GitHub/BrainTumor/predictImage.jpg",imgResult)
