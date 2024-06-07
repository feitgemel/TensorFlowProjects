
#from sklearn.utils import shuffle
import tensorflow as tf
#from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import cv2

IMAGE_SIZE = 128
BATCH_SIZE = 32

# load the images into dataset

# dataset - More than 25K images
#https://www.kaggle.com/puneet6060/intel-image-classification

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "e:/Data-sets/Intel-images/seg_train/seg_train",
    shuffle=True,
    image_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE 
)

class_names = dataset.class_names


print("class_names: " + str(class_names))


# # lets look inside the dataet

for image_batch , label_batch in dataset.take(1): # read info from the first batch (= 32 images and lables )
    print(image_batch.shape)
    print(label_batch.numpy()[1])

# lets see the images in tensor format :
for image_batch , label_batch in dataset.take(1): # read info from the first batch (= 32 images and lables )
    print(image_batch[0])

# lets convert it to numpy 
for image_batch , label_batch in dataset.take(1): # read info from the first batch (= 32 images and lables )
    print(image_batch[0].numpy())


#lets visual one image :
print ("visual an image")
print ("***************")
for image_batch , label_batch in dataset.take(1): # read info from the first batch (= 32 images and lables )
    img = image_batch[0].numpy().astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    classNumber = label_batch.numpy()[0]
    ClassText = class_names[classNumber]

    image = cv2.putText(img ,ClassText, (10,50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,0,0), 1, cv2.LINE_AA )
    cv2.imshow("Image", image)
    cv2.waitKey(0)


# lets visual some images inside a batch

for image_batch , label_batch in dataset.take(1): # read info from the first batch (= 32 images and lables )
    for i in range (10):
    
        img = image_batch[i].numpy().astype("uint8")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        classNumber = label_batch.numpy()[i]
        ClassText = class_names[classNumber]

        image = cv2.putText(img ,ClassText, (10,50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,0,0), 1, cv2.LINE_AA )
        cv2.imshow("Image", image)
        cv2.waitKey(0)

