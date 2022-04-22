from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import cv2

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 25

# load the train images into dataset

Train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/Python-cannot-upload-to-GitHub/Intel-images/seg_train/seg_train",
    shuffle=True,
    image_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE 
)

class_names = Train_dataset.class_names
print("class_names: " + str(class_names))


numberOfClasses = len(class_names)
print("number of classes : " + str(numberOfClasses))

# load the validation images into dataset
Validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/Python-cannot-upload-to-GitHub/Intel-images/seg_test/seg_test",
    shuffle=True,
    image_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE 
)

# reshffule for improve performence


Train_dataset = Train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
Validation_dataset = Validation_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# prepare the data ( resize and normalize )
# build a layer that we will use it later

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling( 1.0 / 255)
])

# layer for data augmentation 

data_agumentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])


input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)



# The model :

model = models.Sequential ([
    resize_and_rescale, # the fitst layer will be resize and normalize
    data_agumentation , # second layer will be data augmentaion based on flip and totate
    layers.Conv2D(16 , (3,3), activation='relu', input_shape= input_shape),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(32 , (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64 , (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(numberOfClasses, activation='softmax')
])

model.build(input_shape = input_shape )

print( model.summary() )

model.compile (
    optimizer = "adam",
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# train the model 
history = model.fit(Train_dataset , epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_data=Validation_dataset)

# print the epochs value of the accuracy 
print(history.history['accuracy'])

# plot the accuracy and loss 

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS), acc , label = "Train accuracy")
plt.plot(range(EPOCHS), val_acc , label = "Validation accuracy")
plt.legend(loc="lower right")
plt.title("Train and validation accuracy ")

plt.subplot(1,2,2)
plt.plot(range(EPOCHS), loss , label = "Train loss")
plt.plot(range(EPOCHS), val_loss , label = "Validation loss")
plt.legend(loc="upper right")
plt.title("Train and validation loss ")


plt.show()

model.save("C:/Python-cannot-upload-to-GitHub/Intel-images/MyModel.h5")