import numpy as np

print("Load the data :")
train_images = np.load("e:/temp/5-vehicles-train-images.npy")
train_labels = np.load("e:/temp/5-vehicles-train-labels.npy")
validatie_images = np.load("e:/temp/5-vehicles-validate-images.npy")
validate_lables = np.load("e:/temp/5-vehicles-validate-labels.npy")
print("Finish load the data .............")

print(train_images.shape)
print(train_labels.shape)
print(validatie_images.shape)
print(validate_lables.shape)

# encode the labels from text to integers :

print(train_labels)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

print(train_labels_encoded)

le.fit(validate_lables)
validate_labels_encoded = le.transform(validate_lables)
print(validate_labels_encoded)

# get the unique categories :
original_labels = le.classes_
print("Unique lables : ")
print(original_labels)

originalLabelsNP = np.array(original_labels)

# just rename the data set :

x_train , y_train , x_test , y_test = train_images, train_labels_encoded, validatie_images, validate_labels_encoded

# normalize the pixe values :

x_train = x_train / 255.0
x_test = x_test / 255.0


# load the VGG16 model without the last layer (The full connected layer )
from keras.applications.vgg16 import VGG16
SIZE = 256

vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

# make the layer non trainable to use the pre-trained weights

for layer in vgg_model.layers:
    layer.trainable = False

print(vgg_model.summary())

#  8 X 8 X 512
# Lets use the features out of the vgg16 model 
# The outcome will be number of train images X 8 X 8 X 512 (This is the last layer of the VGG16 model)

feature_extractor = vgg_model.predict(x_train)

print("Features VGG 16 shape")
print(feature_extractor.shape)

# 5418 images X 8 X 8 X 512 (The last pooling )

# we need to convert the dim to 2D , so the new shape should be :
# Number of images X the rest of all the features 

features = feature_extractor.reshape(feature_extractor.shape[0], -1 )
print("Features after reshape to imagex X features : ")
print(features.shape)
print("**********************************************")

print("Save another copy of the data :")
np.save("e:/temp/5-vehicales-features.npy", features)
np.save("e:/temp//5-vehicales-y_train.npy", y_train)
np.save("e:/temp//5-vehicales-x_test.npy", x_test)
np.save("e:/temp//5-vehicales-y_test.npy", y_test)
print("Finish save the features extraction ")


print("Save the model")
vgg_model.save("e:/temp/5-vehicales-Vgg_model.h5")

print("save categoris :")
np.save("e:/temp/5-vehicales-categories.npy", originalLabelsNP)





