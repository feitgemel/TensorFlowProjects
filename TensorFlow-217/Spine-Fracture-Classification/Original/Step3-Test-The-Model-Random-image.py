import os 
import numpy as np
import tensorflow as tf 
import cv2 
import matplotlib.pyplot as plt
import random

# paths 
#test_img = "D:/Data-Sets-Image-Classification/cervical fracture/test/fracture/CSFDV1B10 (1).png"
model_path = "D:/Data-Sets-Image-Classification/cervical fracture/model.keras"
best_model_path = "D:/Data-Sets-Image-Classification/cervical fracture/best_model.keras"

test_path = "D:/Data-Sets-Image-Classification/cervical fracture/test/" 

IMG_SIZE = (224,224)
class_names = sorted(os.listdir(test_path))
print(class_names)

# load the model
model = tf.keras.models.load_model(best_model_path) # change the path to the "best model" path
print("Model loaded successfully")
print(model.summary())

def predict_random_image():

    random_class = random.choice(class_names)
    class_folder = os.path.join(test_path,random_class)

    # randomly select an image from the class folder
    random_image = random.choice(os.listdir(class_folder))
    image_path = os.path.join(class_folder,random_image)

    # load the image 
    image = cv2.imread(image_path)
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
    plt.title(f"Predicted class : {predicted_class}\nTrue : {random_class}" , fontsize=14)
    plt.axis("off")
    plt.show()


# Run the funcion
predict_random_image()

