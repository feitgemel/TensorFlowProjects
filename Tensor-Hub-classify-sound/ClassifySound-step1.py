import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import io 

# load the model
model = hub.load('https://tfhub.dev/google/yamnet/1')

# the first sound will be silence as mono 16kHz wave format
waveform = np.zeros(3 * 16000 , dtype=np.float32)
print (waveform.shape)

# lets run the model

scores , embeddings ,log_mel_spectogram = model(waveform)



# lets get the model labels 
# =========================

# Extract the path for the lables 
class_map_path = model.class_map_path().numpy()
print ("Class map path :")
print(class_map_path)


class_names = []

with tf.io.gfile.GFile(class_map_path) as csvfile:
    reader = csv.DictReader(csvfile)
    print(reader)

    for row in reader:
        print(row)
        class_names.append(row['display_name'])

print("Class_names : " , class_names)
print(len(class_names))
# 521 classes 

# extract the score
sc = scores.numpy().mean(axis=0)
print(sc)

# get the higher score:
scMax = sc.argmax()
print(scMax)

# lets find the 494 position in the array 
print(class_names[scMax])


