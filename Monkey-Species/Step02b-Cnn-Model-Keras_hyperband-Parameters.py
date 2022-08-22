# pip install tensorflow
# pip install keras_tuner

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint
import numpy as np
import time
import matplotlib.pyplot as plt

# Parameters
IMG=200
IMG_SIZE = [IMG, IMG]

numOfClasses = 10
batchSize = 32
EPOCHS = 50
PATIENCE=5 



# build the model :

def build_model(hp):

    filters_layer1=hp.Int('filters_layer1',min_value=32 , max_value=256, step=32)
    filters_layer2=hp.Int('filters_layer2',min_value=32 , max_value=256, step=32)
    filters_layer3=hp.Int('filters_layer3',min_value=32 , max_value=256, step=32)
    filters_layer4=hp.Int('filters_layer4',min_value=32 , max_value=256, step=32)

    hp_learning_rate=hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4 ])
    hp_optimizer = tf.keras.optimizers.Adam(learning_rate = hp_learning_rate )
    hp_dropout = hp.Choice('drop_out', values=[0.3 , 0.5])

    hp_last_dense_layer = hp.Int('last_dense_layer',min_value=128 , max_value=768, step=64)





    model1 = tf.keras.models.Sequential ([

        tf.keras.layers.Conv2D(filters_layer1,kernel_size=(3,3), activation='relu', input_shape=(IMG,IMG,3)),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(filters_layer2,kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(filters_layer3,kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(filters_layer4,kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(hp_dropout),

        tf.keras.layers.Dense(hp_last_dense_layer, activation='relu'),

        tf.keras.layers.Dense(numOfClasses, activation='softmax')


    ])

    model1.compile(loss='categorical_crossentropy', optimizer=hp_optimizer , metrics=['accuracy'])

    return model1


#model = build_model()

#print(model.summary())


# compile
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Data
# =====

trainMyImagesFolder = "C:/Python-cannot-upload-to-GitHub/10-Monkey-Species/training/training"
testMyImagesFolder = "C:/Python-cannot-upload-to-GitHub/10-Monkey-Species/validation/validation"

train_datagen = ImageDataGenerator(rescale = 1. / 255, 
                                    rotation_range = 20 ,
                                    width_shift_range = 0.2 ,
                                    height_shift_range = 0.2 ,
                                    shear_range = 0.2 ,
                                    zoom_range = 0.2 ,
                                    horizontal_flip = True)

training_set = train_datagen.flow_from_directory(trainMyImagesFolder,
                                                shuffle=True,
                                                target_size=IMG_SIZE,
                                                batch_size=batchSize,
                                                class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1. / 255)


test_set = test_datagen.flow_from_directory(testMyImagesFolder,
                                                shuffle=False, #### important
                                                target_size=IMG_SIZE,
                                                batch_size=batchSize,
                                                class_mode = 'categorical')


stepsPerEpochs = np.ceil (training_set.samples / batchSize) # round the result up
validationSteps =np.ceil (test_set.samples / batchSize) 

best_model_file = "C:/Python-cannot-upload-to-GitHub/10-Monkey-Species/myCnnMonkeyModelHyperBandParam.h5"
#bestModel = ModelCheckpoint(best_model_file, monitor='val_accuracy', verbose=1, save_best_only=True)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE)



#keras tuner -> dont forget to pip install keras_tuner
import keras_tuner
from keras_tuner import RandomSearch, Hyperband

# tuner = RandomSearch(
#     build_model , 
#     objective='val_accuracy',
#     max_trials=5 , # how many random options
#     executions_per_trial=12, # in each version , how many times to execute it
#     directory="c:/temp",
#     project_name='MoneyCnnRandomSearch',
#     overwrite=True    )


tuner = Hyperband (
    build_model, # our function name
    objective='val_accuracy',
    max_epochs=100, # its like number of tourments
    factor=3, # in each version , how many time to execute it
    directory ='c:/temp',
    project_name = 'MoneyCnnHyperSearch',
    overwrite=True
    )


# fit the model

# history = model.fit (
#     training_set,
#     validation_data = test_set,
#     epochs = EPOCHS,
#     steps_per_epoch = stepsPerEpochs,
#     validation_steps = validationSteps,
#     verbose=1,
#     callbacks=[bestModel])


# similar to fit
tuner.search(
    training_set,
    validation_data = test_set,
    epochs = EPOCHS,
    batch_size = batchSize,
    callbacks=[stop_early],
    steps_per_epoch=stepsPerEpochs,
    validation_steps=validationSteps   )




# let's get the resutls :
# What is our best paramters :

best_hp = tuner.get_best_hyperparameters()[0].values 
print("==================================")
print("Best model parameters :")
print(best_hp) 
print("==================================")
print("  ")

# What is our best model  :

model = tuner.get_best_models(num_models=1)[0]
print("==================================")
print("Best model is  :")
print(model.summary())
print("==================================")
print("  ")

model.save(best_model_file)

