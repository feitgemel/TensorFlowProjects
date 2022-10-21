# pip install --upgrade tf-keras-vis tensorflow

import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16 as Vgg16Model

model = Vgg16Model(weights='imagenet', include_top=True)
print (model.summary())

# define a function to modify the model
# we will change the softmax to a linear function

def model_modifier(modl):
    modl.layers[-1].activation = tf.keras.activations.linear # All the layers except the last one will have activation=linear


# create an object instance of Acticvation maximization class

from tf_keras_vis.activation_maximization import ActivationMaximization

activation_maximization = ActivationMaximization(model,
                                                model_modifier,
                                                clone=True) # clone means , duplicate the model and not update the current one

# Now , We will define a loss function that maximize a specific class.
# leats maximize the class of "Persian cat" , class no. 283

# https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/

def loss(output):
    return output[:, 283]


# visual the class

from tf_keras_vis.utils.callbacks import Print

activation = activation_maximization(loss,
                                    callbacks=[Print(interval=50)]    )

#lets grab the image after running the process
image = activation[0].astype(np.uint8)

# show the image using OpenCv
import cv2

# change the image from RGB to BGR

imageCV = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)


# enlarge the image
scale_percent = 200
w = int(imageCV.shape[1]* scale_percent / 100)
h = int(imageCV.shape[0]* scale_percent / 100)
dim = (w, h)

resized = cv2.resize(imageCV, dim , interpolation=cv2.INTER_AREA)


cv2.imshow("Persian Cat",resized )
cv2.waitKey(0)





