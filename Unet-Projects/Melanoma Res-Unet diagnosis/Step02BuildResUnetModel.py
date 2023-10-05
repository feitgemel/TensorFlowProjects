from tensorflow.keras.layers import Conv2D, BatchNormalization , Activation, MaxPool2D, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Model 

def batchnorm_relu(inputs):
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    return x

def residual_block(inputs, num_filters, strides=1):
    # Conv layer
    x = batchnorm_relu(inputs)
    x = Conv2D(num_filters, 3, padding="same", strides=strides)(x) 
    x = batchnorm_relu(x)
    x = Conv2D(num_filters, 3, padding="same", strides=1)(x)

    # shortcut connection

    s = Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)
    x = x + s
    return x


def decoder_block(inputs, skip_features , num_filters):
    x = UpSampling2D((2,2))(inputs)
    x = Concatenate()([x, skip_features])
    x = residual_block(x, num_filters, strides=1)
    return x


# The main function
def build_resunet(input_shape):

    inputs = Input(input_shape)

    # Encoder 1 - First block 
    x = Conv2D(64, 3, padding="same", strides=1)(inputs)
    x = batchnorm_relu(x)
    x = Conv2D(64, 3, padding="same", strides=1)(x)
    s = Conv2D(64,1 , padding="same", strides=1)(inputs) # this is the sortcut
    s1 = x + s

    # Encoder 2 and 3 - Block 2 and 3
    s2 = residual_block(s1, 128, strides=2) # the strides = 2
    s3 = residual_block(s2, 256, strides=2 ) # the strides = 2

    # the bridge 
    b = residual_block(s3 , 512, strides=2)

    # decoder 1 , 2, 3 

    d1 = decoder_block(b, s3, 256)
    d2 = decoder_block(d1, s2, 128)
    d3 = decoder_block(d2, s1 , 64)

    # Classifier 
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d3)

    # THE MODEL
    model = Model(inputs, outputs)
    return model


if __name__ == "__main__":
    model = build_resunet((256,256,3 ))
    print(model.summary())










