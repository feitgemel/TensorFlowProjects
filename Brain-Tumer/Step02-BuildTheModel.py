from matplotlib.pyplot import cla
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, MaxPool2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.preprocessing import image
import keras



TRAIN_DIR = "C:/Python-cannot-upload-to-GitHub/BrainTumor/train"
VAL_DIR = "C:/Python-cannot-upload-to-GitHub/BrainTumor/validate"

# build the CNN model
# ===================

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape = (224,224,3)    ))

model.add(Conv2D(filters=64, kernel_size=(3,3),  activation='relu' ))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3),  activation='relu' ))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3),  activation='relu' ))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(rate=0.25))

model.add(Flatten())

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(rate=0.25))

#final layer:
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

print(model.summary())


# create the train data augmentation object
# ==========================================

train_datagen = image.ImageDataGenerator(
    zoom_range=0.2 , shear_range=0.2, rescale=1. / 255 , horizontal_flip=True
)

val_datagen = image.ImageDataGenerator( rescale= 1. / 255)

train_data = train_datagen.flow_from_directory(directory=TRAIN_DIR , target_size=(224,224) , batch_size=32 , class_mode='binary')
val_data = val_datagen.flow_from_directory(directory=VAL_DIR, target_size=(224,224), batch_size=32, class_mode='binary')



# create model check point for the performence of the model

from keras.callbacks import ModelCheckpoint , EarlyStopping


# lets stop the training if the accuracy is good

es = EarlyStopping(monitor='val_accuracy', min_delta=0.01 , patience=5 , verbose=1 , mode='auto')
mc = ModelCheckpoint(filepath='C:/Python-cannot-upload-to-GitHub/BrainTumor/MyBestModel.h5', monitor='val_accuracy' ,  verbose=1 , mode='auto' , save_best_only=True)


call_back = [es, mc]

hist = model.fit(x=train_data, epochs=30 , verbose=1, validation_data=val_data, callbacks=call_back)

h = hist.history
print('Keys : ', h.keys() )

#lets plot the accuracy and the loss
#===================================


#accuracy 
plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'], c='red')
plt.title('Accuracy vs. Val Accuracy')
plt.show()

#loss
plt.plot(h['loss'])
plt.plot(h['val_loss'], c='red')
plt.title('loss vs. Val loss')
plt.show()