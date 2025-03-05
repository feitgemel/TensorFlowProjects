import os 
import numpy as np
import tensorflow as tf 
import cv2 
import matplotlib.pyplot as plt

# paths 
test_img = "D:/Data-Sets-Image-Classification/cervical fracture/test/fracture/CSFDV1B10 (1).png"
model_path = "D:/Data-Sets-Image-Classification/cervical fracture/model.keras"
test_path = "D:/Data-Sets-Image-Classification/cervical fracture/test/" 

IMG_SIZE = (224,224)
class_names = sorted(os.listdir(test_path))
print(class_names)

# load the model
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully")
print(model.summary())


image = cv2.imread(test_img)
image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# process the image :
resized_image = cv2.resize(image_rgb,IMG_SIZE)
# normalize the image :
normalized_image = resized_image/255.0

print(normalized_image.shape) # -> (224, 224, 3)
# expand the dimensions : (224,224,3) -> (1,224,224,3) - create a batch of 1 image
input_array = np.expand_dims(normalized_image,axis=0)
print(input_array.shape) # (1, 224, 224, 3) 

# predict the image :
predictions = model.predict(input_array)
print(predictions)
predicted_class_index = np.argmax(predictions)

print("Predicted class index : ",predicted_class_index)

predicted_class = class_names[predicted_class_index]
print("Predicted class Name : ",predicted_class)

# display the image :
plt.figure(figsize=(6,6))
plt.imshow(image_rgb)
plt.title(f"Predicted class : {predicted_class}\nTrue : fracture:" , fontsize=14)
plt.axis("off")
plt.show()










#cv2.imshow("Image",image)
#cv2.waitKey(0)


