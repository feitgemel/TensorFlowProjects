# pip install tensorflow==2.10
# pip install pandas numpy matplotlib opencv-python librosa

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import cv2
import os
from glob import glob
import librosa

# lets load one wav file using librosa

y, sr = librosa.load("C:/Data-Sets/Musical Chord Classification/Audio_Files/Major/Major_6.wav")

# y = the raw data of the audio file
# sr = sample rate

print("10 first values of the audio")
print(y[:10])

print("The shape of the audio : ")
print(y.shape)

print("Sample rate :" + str(sr))

# convert one audio file to spectogram (Visual of the audio)

D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) # convert from Amplitue to decibel

print(S_db.shape)
print(type(S_db))

# convert it to "image" values

audioAsImage = (S_db * 255).astype(np.uint8)
print(audioAsImage)
print(audioAsImage.shape)

# save the image
cv2.imwrite("e:/temp/Major_6.wav.png",audioAsImage)

# transform all the Major and Minor audio files to images

# Major files
Path_for_Major_Spectogram="C:/Data-Sets/Musical Chord Classification/New/Major"
isExist = os.path.exists(Path_for_Major_Spectogram)
if not isExist:
    os.makedirs(Path_for_Major_Spectogram)
    print("The new Major folder created !")

MajorAudioFiles = glob("C:/Data-Sets/Musical Chord Classification/Audio_Files/Major/*.wav")

for file in MajorAudioFiles:
    y , sr = librosa.load(file)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D),ref=np.max)
    ImageAudio = (S_db * 255).astype(np.uint8)

    # extract the file name out of the full name
    idx = file.rfind("\\")
    filename = file[idx+1:]

    cv2.imwrite(Path_for_Major_Spectogram+"/"+filename+".png",ImageAudio)
    print(filename)


# Minor files
Path_for_Minor_Spectogram="C:/Data-Sets/Musical Chord Classification/New/Minor"
isExist = os.path.exists(Path_for_Minor_Spectogram)
if not isExist:
    os.makedirs(Path_for_Minor_Spectogram)
    print("The new Minor folder created !")

MinorAudioFiles = glob("C:/Data-Sets/Musical Chord Classification/Audio_Files/Minor/*.wav")

for file in MinorAudioFiles:
    y , sr = librosa.load(file)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D),ref=np.max)
    ImageAudio = (S_db * 255).astype(np.uint8)

    # extract the file name out of the full name
    idx = file.rfind("\\")
    filename = file[idx+1:]

    cv2.imwrite(Path_for_Minor_Spectogram+"/"+filename+".png",ImageAudio)
    print(filename + " - Shape : "+ str(ImageAudio.shape))


#test one saved file 
img = cv2.imread("C:/Data-Sets/Musical Chord Classification/New/Minor/Minor_35.wav.png",0)
cv2.imshow("img",img)
print(img.shape)
cv2.waitKey(0)






