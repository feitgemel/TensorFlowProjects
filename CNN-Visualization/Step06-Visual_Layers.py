from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = 200
BatchSize = 32

# load out model 
modelPath = "C:/Python-cannot-upload-to-GitHub/10-Monkey-Species/myTransferLearningMonkeyModel.h5"

model = load_model(modelPath)
print(model.summary())

AllLayers = model.layers

print("List of the layers")
print(AllLayers)

# lets create a list of the layers

print("Total layers : " + str(len(AllLayers)))
for count, layer in enumerate(AllLayers):
    print("layer no. "+ str(count)+ " : "+ layer.name)

# Total layers : 21
# layer no. 0 : input_1
# layer no. 1 : block1_conv1
# layer no. 2 : block1_conv2
# layer no. 3 : block1_pool
# layer no. 4 : block2_conv1
# layer no. 5 : block2_conv2
# layer no. 6 : block2_pool
# layer no. 7 : block3_conv1
# layer no. 8 : block3_conv2
# layer no. 9 : block3_conv3
# layer no. 10 : block3_pool
# layer no. 11 : block4_conv1
# layer no. 12 : block4_conv2
# layer no. 13 : block4_conv3
# layer no. 14 : block4_pool
# layer no. 15 : block5_conv1
# layer no. 16 : block5_conv2
# layer no. 17 : block5_conv3
# layer no. 18 : block5_pool
# layer no. 19 : flatten
# layer no. 20 : dense

# lets plot the oucome of the layers
# we will create a new short model with our visual relevant layers

conv_layer_index = [1,2,4]
outputs=[]

for i in conv_layer_index:
    print(i)
    outputs.append(model.layers[i].output)

print(outputs)

from keras.models import Model
from tensorflow.keras.preprocessing import image

model_short = Model(inputs=model.inputs, outputs=outputs)
print(model_short.summary())

# load an image
imagePath = "C:/Python-cannot-upload-to-GitHub/10-Monkey-Species/validation/validation/n3/n311.jpg"

img = image.load_img(imagePath, target_size=(IMG_SIZE,IMG_SIZE))
imgNp = image.img_to_array(img)
imgNp = imgNp / 255.0

print(imgNp.shape)
imgToModel = np.expand_dims(imgNp, axis = 0)
print(imgToModel.shape)

# lets run the "prediction" to get the outpus of the filters
feature_output = model_short.predict(imgToModel) 

feature_one_layer = feature_output[1] # get the result of layer 1
print(feature_one_layer.shape) # Get how many filters for layer 1 ?

# lets see outcome of filter no. 20 (out of 64 filters) in layer 1
plt.imshow(feature_one_layer[0][:,:,20])
plt.show()


# layer 2 - show outcome
feature_one_layer = feature_output[2] # get the result of layer 2
print(feature_one_layer.shape) # Get how many filters for layer 2 ?

# lets see outcome of filter no. 20 (out of 64 filters) in layer 1
plt.imshow(feature_one_layer[0][:,:,20])
plt.show()


# lets plot all 64 outcome filters of our image (of all model short layers : 1 , 2 , 4)

cols=8
rows=8
layer_index = 0

for ftr in feature_output:
    fig= plt.figure(figsize=(12,12))
    layerDisplayNumber = conv_layer_index[layer_index]
    fig.suptitle("Layer number : "+str(layerDisplayNumber), fontsize=20 )

    for i in range(1, cols*rows + 1):
        fig = plt.subplot(rows, cols , i)
        fig.set_xticks([])
        fig.set_yticks([])

        plt.imshow(ftr[0, :, :, i-1])
        
    plt.show()
    layer_index = layer_index + 1
        
