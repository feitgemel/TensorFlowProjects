import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# convolutional block 
def conv_block(x , num_filters):
    x = Conv2D(num_filters , (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters , (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)  

    return x

# build the model

def build_model(shape):
    num_filters = [16,32,48,64] # in the original Unet : 64 , 128 , 256, 512
    inputs = Input((shape))

    skip_x = []
    x = inputs

    # Encoder

    for f in num_filters:
        x = conv_block(x , f)
        skip_x.append(x)
        x = MaxPool2D((2,2))(x)

    # Bridge 
    x = conv_block(x, 128) # in the original Unet it should be 1024


    # decoder 
    num_filters.reverse()
    skip_x.reverse()

    for i , f in enumerate(num_filters):
        x = UpSampling2D((2,2))(x)
        xs = skip_x[i]
        x = Concatenate()([x,xs])
        x = conv_block(x,f)
    
    ## output layer :
    x = Conv2D(1, (1,1), padding="same")(x)
    x = Activation("sigmoid")(x) # binary classification

    return Model(inputs, x)




