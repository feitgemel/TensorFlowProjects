import numpy as np
import cv2
import sounddevice as sd
import librosa
import tensorflow as tf

best_model_file = "e:/temp/Audio-Mijor-Minor.h5"
model = tf.keras.models.load_model(best_model_file) 
print(model.summary())

shape=(97,1025)

from tensorflow.keras.utils import img_to_array 

def prepareAudio(pathForAudio):
    y , sr = librosa.load(pathForAudio)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D),ref=np.max)
    ImageAudio = (S_db * 255).astype(np.uint8)

    resizedImage = cv2.resize(ImageAudio,shape,interpolation=cv2.INTER_AREA)

    imgResult = img_to_array(resizedImage)
    imgResult = np.expand_dims(imgResult, axis=0)
    imgResult = imgResult / 255.

    return imgResult


# run the prediction
#testAudio = "C:/Data-Sets/Musical Chord Classification/Audio_Files/Major/Major_11.wav"
testAudio = "C:/Data-Sets/Musical Chord Classification/Audio_Files/Minor/Minor_14.wav"




audio_file , sr = librosa.load(testAudio)
sd.play(audio_file, sr)
sd.wait()

imageForModel = prepareAudio(testAudio)
resultArray = model.predict(imageForModel, verbose=1)
print(resultArray)

answer = resultArray[0][0]

if answer < 0.5:
    print("Major chord")
else :
    print("Minor chord")