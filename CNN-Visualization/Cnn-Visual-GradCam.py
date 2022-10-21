# pip install --upgrade tf-keras-vis tensorflow
# =============================================

from xml.etree.ElementInclude import include
import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16 as Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img


#load the Vgg16 model
model = Model(weights='imagenet',include_top=True)
print(model.summary())

# 283	Persian cat
# 150	sea lion

# here is the link for imagenet classes : https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/

image_titles = ['Persian-cat','Sea-lion']

# load the images

img1=load_img("C:/GitHub/TensorFlowProjects/CNN-Visualization/Persian-cat.jpg", target_size=(224,224))
img2=load_img("C:/GitHub/TensorFlowProjects/CNN-Visualization/Sea-lion.jpg", target_size=(224,224))

print(type(img1))

# convert it to Numpy Array:
img1_array = cv2.cvtColor(np.array(img1),cv2.COLOR_RGB2BGR)
img2_array = cv2.cvtColor(np.array(img2),cv2.COLOR_RGB2BGR)

# Show the images 

cv2.imshow("Original - Cat ", img1_array)
cv2.imshow("Original - Sea-lion", img2_array)
cv2.waitKey(0)


# prepare the data for the Vgg16 model
images = np.asarray([np.array(img1), np.array(img2)])
X = preprocess_input(images)

# define the loss functions with a traget classes 
def loss(output):
    return(output[0][283], output[1][150])


# define the model modifier - change the activation function
def model_modifier(mdl):
    mdl.layers[-1].activation = tf.keras.activations.linear # we change the activation function of last layer to linear


# define the grand cam function
from tf_keras_vis.utils import normalize
from matplotlib import cm 
from tf_keras_vis.gradcam import Gradcam

# create an object

gradcam = Gradcam(model,
                model_modifier=model_modifier,
                clone=False)


cam = gradcam(loss, X , penultimate_layer=-1 )# the layer befor the softmax

cam = normalize(cam)

# lets show the outcome :

# to extract the image from the model 

heatmapImg1 = np.uint8(cm.jet(cam[0])[..., :3] * 255 )
# chnage the color map to jet
heatmapImg1 = cv2.applyColorMap(heatmapImg1 , cv2.COLORMAP_JET)

# lets add some alpha transparency
alpha = 0.5
overlay = heatmapImg1.copy() # copy the image
result1 = cv2.addWeighted(img1_array, alpha, heatmapImg1 , 1-alpha, 0)

scale_precent = 200
w = int(heatmapImg1.shape[1] * scale_precent / 100)
h = int(heatmapImg1.shape[0] * scale_precent / 100)
dim = (w,h)

result1 = cv2.resize(result1, dim , interpolation=cv2.INTER_AREA)
img1_array = cv2.resize(img1_array, dim , interpolation=cv2.INTER_AREA)

cv2.imshow("GradCam - Cat",result1 )
#cv2.imwrite("GradCam - Cat.jpg",result1 )
cv2.imshow("Original - Cat",img1_array )
cv2.waitKey(0)

# lets show the sea lion

heatmapImg2 = np.uint8(cm.jet(cam[1])[..., :3] * 255 )
heatmapImg2 = cv2.applyColorMap(heatmapImg2 , cv2.COLORMAP_JET)
overlay = heatmapImg2.copy() # copy the image
result2 = cv2.addWeighted(img2_array, alpha, heatmapImg2 , 1-alpha, 0)

w = int(heatmapImg2.shape[1] * scale_precent / 100)
h = int(heatmapImg2.shape[0] * scale_precent / 100)
dim = (w,h)

result2 = cv2.resize(result2, dim , interpolation=cv2.INTER_AREA)
img2_array = cv2.resize(img2_array, dim , interpolation=cv2.INTER_AREA)

cv2.imshow("GradCam - Sea lion",result2 )
cv2.imshow("Original - Sea lion",img2_array )
#cv2.imwrite("GradCam - Sea-lion.jpg",result2 )
cv2.waitKey(0)

