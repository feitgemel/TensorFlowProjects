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

# lets look at the filters of layer no. 1
print("===================================================")
print("Conv 1")
filters1 , biases = model.layers[1].get_weights()
print(AllLayers[1].name)
print(filters1.shape)

# show filyer no. 20 out of 64
plt.imshow(filters1[:,:,0,20], cmap='gray') # show filter no 20 . 0 is the Red channal 
plt.show()
print("===================================================")
print("Conv 2")
filters2 , biases = model.layers[2].get_weights()
print(AllLayers[2].name)
print(filters1.shape)

# show filyer no. 20 out of 64
plt.imshow(filters2[:,:,0,20], cmap='gray') # show filter no 20 . 0 is the Red channal 
plt.show()

print("===================================================")
print("Conv 4")
filters4 , biases = model.layers[4].get_weights()
print(AllLayers[4].name)
print(filters1.shape)

# show filyer no. 20 out of 64
plt.imshow(filters4[:,:,0,20], cmap='gray') # show filter no 20 . 0 is the Red channal 
plt.show()

# lets see the whole 64 filters of layer 1

# plot the filters :
fig1 = plt.figure(figsize=(8,12))
fig1.suptitle("Display filters of layer no. 1 ", fontsize=20)

cols=8
rows=8 
n_filters = cols * rows # total 64 filters , 8 per rows in 8 columns

for i in range(1, n_filters + 1):
    f = filters1[:,:,:, i-1]
    fig1 = plt.subplot(rows, cols, i)
    fig1.set_xticks([]) # turn of the axis
    fig1.set_yticks([]) # turn of the axis
    plt.imshow( f[:,:,0], cmap='gray') # show the filters of the Red channal = 0

plt.show()
