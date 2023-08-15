import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model 

# convolutional block
def conv_block(x , num_filters) :
    x = Conv2D(num_filters, (3,3), padding="same")(x)
    x= BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x= Activation("relu")(x)

    return x


# build the model

def build_model(shape):

    # original sizes : 64, 128, 256, 512
    num_filters = [16,32,48,64]

    inputs = Input((shape))

    skip_x = []
    x= inputs

    # encoder Unet part 

    for f in num_filters:
        x = conv_block(x , f)
        skip_x.append(x)
        x= MaxPool2D((2,2))(x)

    # bridge with 1024 filters 
    x = conv_block(x, 128)


    # prepare for the decoder
    num_filters.reverse()
    skip_x.reverse()

    # Decoder Unet part
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2,2))(x)
        xs = skip_x[i]
        x = Concatenate()([x,xs])
        x= conv_block(x,f)

    #output
    x = Conv2D(1, (1,1) , padding="same")(x)
    x = Activation("sigmoid")(x) # since it is a binary classification and segmentation

    return Model(inputs,x)








