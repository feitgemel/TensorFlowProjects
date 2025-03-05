# Dataset : https://www.kaggle.com/datasets/msambare/fer2013

import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt

# Paths to train and test data
trainPath = "C:/Data-Sets/Emotion-Faces/train"
testPath = "C:/Data-Sets/Emotion-Faces/test"

# Load folder categories
folderList = sorted(os.listdir(trainPath))
print(folderList)

# Prepare data arrays
X_train, y_train = [], []
X_test, y_test = [], []

# Load training data
for i, category in enumerate(folderList):
    files = os.listdir(os.path.join(trainPath, category))
    for file in files:
        print(f"{category}/{file}")
        img = cv2.imread(os.path.join(trainPath, category, file), cv2.IMREAD_GRAYSCALE)
        X_train.append(img)
        y_train.append(i)  # Assign folder index as label

print(f"Number of training images: {len(X_train)}")

# Load test data
folderList = sorted(os.listdir(testPath))
for i, category in enumerate(folderList):
    files = os.listdir(os.path.join(testPath, category))
    for file in files:
        print(f"{category}/{file}")
        img = cv2.imread(os.path.join(testPath, category, file), cv2.IMREAD_GRAYSCALE)
        X_test.append(img)
        y_test.append(i)

print(f"Number of test images: {len(X_test)}")

# Convert data to numpy arrays
X_train = np.array(X_train, dtype='float32')
y_train = np.array(y_train, dtype='float32')
X_test = np.array(X_test, dtype='float32')
y_test = np.array(y_test, dtype='float32')

# Normalize images to 0-1 range
X_train /= 255.0
X_test /= 255.0

# Reshape to include channel dimension
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# Convert labels to categorical format
num_classes = 7
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(48, 48, 1)),
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

    tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

    tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

    tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

# Train the model
batch_size = 32
epochs = 30
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    shuffle=True,
    callbacks=[early_stopping]
)

# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.plot(acc, label='Training Accuracy', color='r')
plt.plot(val_acc, label='Validation Accuracy', color='b')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.figure()
plt.plot(loss, label='Training Loss', color='r')
plt.plot(val_loss, label='Validation Loss', color='b')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model
model.save("e:/temp/emotion_model.h5")
