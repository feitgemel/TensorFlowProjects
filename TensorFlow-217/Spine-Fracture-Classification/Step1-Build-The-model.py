"""
Spine Fracture Detection using Convolutional Neural Networks (CNN)

Description:
This script implements a CNN model for detecting spine fractures in X-ray images.
The model uses a custom CNN architecture to classify X-ray images into two categories:
fracture and non-fracture cases.

Dataset: Spine Fracture X-ray Dataset from Kaggle
Model Architecture: CNN with multiple convolutional layers, max pooling, and dense layers
Input: X-ray images (224x224x3)
Output: Binary classification (fracture/no fracture)
"""

# Import necessary libraries
import tensorflow as tf  # Main deep learning framework
print(tf.__version__)   # Display TensorFlow version for reproducibility

import numpy as np      # For numerical operations
import pandas as pd     # For data manipulation
import matplotlib.pyplot as plt  # For visualization

from keras.utils import img_to_array, load_img  # Utilities for image processing

# Dataset path configuration
train_path = "D:/Data-Sets-Image-Classification/cervical fracture/train/"  # Path to training data
valid_path = "D:/Data-Sets-Image-Classification/cervical fracture/val/"    # Path to validation data

# Model configuration parameters
BATCH_SIZE = 32            # Number of images processed in each training iteration
IMG_SIZE = (224,224)       # Input image dimensions (height, width)
IMG_DIM = (224,224,3)      # Input image dimensions with channels (height, width, channels)
EPOCHS = 25               # Number of complete passes through the training dataset
NUM_CLASSES = 2           # Number of classification categories (fracture/no fracture)

# Load and display a sample image for verification
img = load_img(train_path + "fracture/CSFDV1B10 (18)-sharpened-rotated3.png")
plt.imshow(img)           # Display the image
img = img_to_array(img)   # Convert image to numpy array
print(img.shape)          # Display image dimensions
plt.show()

# Data loading and preprocessing
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    shuffle = True,        # Shuffle data for better training
    label_mode = 'int'     # Use integer labels
)

valid_dataset = tf.keras.utils.image_dataset_from_directory(
    valid_path,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    shuffle = True,
    label_mode = 'int'
)

# Normalization function: Convert pixel values from [0-255] to [0-1] range
def normalize_image(image, label):
    return tf.cast(image/255.0, tf.float32), label

# Apply normalization to datasets
train_dataset = train_dataset.map(normalize_image)
valid_dataset = valid_dataset.map(normalize_image)

# Import required Keras components for model building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPool2D, Conv2D, Dropout
from tensorflow.keras.optimizers import Adam

# Configure optimizer with learning rate
optimizer = Adam(learning_rate = 0.001)

# Define CNN model architecture
def get_cnn_model():
    model = Sequential()
    # First convolutional block
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=IMG_DIM))
    model.add(MaxPool2D(3,3))
    
    # Second convolutional block
    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(3,3))
    
    # Third convolutional block
    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(Flatten())  # Flatten the 3D output to 1D for dense layers
    
    # Dense layers for classification
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))  # Prevent overfitting
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))  # Output layer
    
    # Compile model with specified optimizer and loss function
    model.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

# Create and display model summary
model = get_cnn_model()
print(model.summary())

# Train the model
hist = model.fit(
    train_dataset,
    validation_data = valid_dataset,
    epochs = EPOCHS
)

# Save the trained model
model.save("D:/Data-Sets-Image-Classification/cervical fracture/model.keras")

# Visualization of training results
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs_range = range(EPOCHS)

# Create subplot for accuracy and loss visualization
plt.figure(figsize=(12,6))

# Plot training and validation accuracy
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot training and validation loss
plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()