import tensorflow as tf
print(tf.__version__)

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from keras.utils import img_to_array , load_img

# Dataset : https://www.kaggle.com/datasets/vuppalaadithyasairam/spine-fracture-prediction-from-xrays

train_path = "D:/Data-Sets-Image-Classification/cervical fracture/train/" # dont forget to add / at the end
valid_path = "D:/Data-Sets-Image-Classification/cervical fracture/val/" # dont forget to add / at the end

BATCH_SIZE = 32
IMG_SIZE = (224,224) # 224,224 is the input size for the model: Image dimensions
IMG_DIM = (224,224,3) # 224,224,3 is the input size for the model WITH 3 channels
EPOCHS = 500 # 25
NUM_CLASSES = 2


# Display a sample image
img = load_img(train_path + "fracture/CSFDV1B10 (18)-sharpened-rotated3.png")
plt.imshow(img)
img = img_to_array(img)
print(img.shape)
plt.show()

# Load the data set :

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    shuffle = True,
    label_mode = 'int')

valid_dataset = tf.keras.utils.image_dataset_from_directory(
    valid_path,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    shuffle = True,
    label_mode = 'int')

# normalize the data : convert the image pixel values to from 0-255 values to 0 -> 1 range
def normalize_image(image,label):
    return tf.cast(image/255.0,tf.float32),label

# convert the data :
train_dataset = train_dataset.map(normalize_image)
valid_dataset = valid_dataset.map(normalize_image)

# Build the model :

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Flatten, MaxPool2D, Conv2D , Dropout
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate = 0.001)

def get_cnn_model():
    model = Sequential()
    model.add(Conv2D( 32 , kernel_size = 3 , padding = 'same' , activation = 'relu' , input_shape = IMG_DIM))
    model.add(MaxPool2D(3,3))
    model.add(Conv2D( 64 , kernel_size = 3 , padding = 'same' , activation = 'relu'))
    model.add(MaxPool2D(3,3))
    model.add(Conv2D( 128 , kernel_size = 3 , padding = 'same' , activation = 'relu'))
    model.add(Flatten())

    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation = 'relu'))   
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation = 'softmax'))

    model.compile(optimizer = optimizer , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])

    return model

model = get_cnn_model()
print(model.summary())



from keras.callbacks import ModelCheckpoint, EarlyStopping

# save the best model during training
checkpoint_path = "D:/Data-Sets-Image-Classification/cervical fracture/best_model.keras"
checkpoint_callback = ModelCheckpoint(
    filepath = checkpoint_path,
    monitor="val_loss",
    save_best_only = True,
    verbose = 1)

# Early stopping 
early_stop = EarlyStopping(monitor = 'val_loss',patience = 20,  verbose = 1)


# Train the model :
hist = model.fit(
    train_dataset,
    validation_data = valid_dataset,
    epochs = EPOCHS,
    callbacks=[early_stop, checkpoint_callback]
)


# Save the final model :
model.save("D:/Data-Sets-Image-Classification/cervical fracture/model.keras")

# Plot the resutls :
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs_range = range(len(acc)) # the rnage of the digaram is not the maximum epoch value (500) but the actual number of epochs 

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.plot(epochs_range, acc , label = 'Training Accuracy')
plt.plot(epochs_range, val_acc , label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss , label = 'Training Loss')
plt.plot(epochs_range, val_loss , label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')

plt.show()






