from fileinput import filename
import os
import random
import shutil

splitSize = .85

# show the list of folders
dataDirList = os.listdir("C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/Chess")
print(dataDirList)

# lest vuild a function that will split the data between train and validation

def split_data(SOURCE , TRAINING , VALIDATION , SPLIT_SIZE):

    files = []

    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        print(file)
        if os.path.getsize(file) > 0 :
            files.append(filename)
        else:
            print(filename + " - would ignore this file")

    print(len(files))

    trainLength = int( len(files) * SPLIT_SIZE)
    validLength = int (len(files) - trainLength)
    shuffledSet = random.sample(files , len(files))

    trainSet = shuffledSet[0:trainLength]
    validSet = shuffledSet[trainLength:]

    # copy the train images :
    for filename in trainSet:
        thisfile = SOURCE + filename
        destination = TRAINING + filename
        shutil.copyfile(thisfile, destination)

    # copy the validation images :
    for filename in validSet:
        thisfile = SOURCE + filename
        destination = VALIDATION + filename
        shutil.copyfile(thisfile, destination)

BishopSourceDir = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/Chess/Bishop/" #dont forget the last "/"
BishopTrainDir = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/train/Bishop/" #dont forget the last "/"
BishopValDir = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/validation/Bishop/" #dont forget the last "/"

KingSourceDir = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/Chess/King/" #dont forget the last "/"
KingTrainDir = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/train/King/" #dont forget the last "/"
KingValDir = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/validation/King/" #dont forget the last "/"

KnightSourceDir = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/Chess/Knight/" #dont forget the last "/"
KnightTrainDir = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/train/Knight/" #dont forget the last "/"
KnightValDir = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/validation/Knight/" #dont forget the last "/"

PawnSourceDir = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/Chess/Pawn/" #dont forget the last "/"
PawnTrainDir = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/train/Pawn/" #dont forget the last "/"
PawnValDir = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/validation/Pawn/" #dont forget the last "/"

QueenSourceDir = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/Chess/Queen/" #dont forget the last "/"
QueenTrainDir = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/train/Queen/" #dont forget the last "/"
QueenValDir = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/validation/Queen/" #dont forget the last "/"

RookSourceDir = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/Chess/Rook/" #dont forget the last "/"
RookTrainDir = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/train/Rook/" #dont forget the last "/"
RookValDir = "C:/Python-cannot-upload-to-GitHub/Chessman-image-dataset/validation/Rook/" #dont forget the last "/"

split_data(BishopSourceDir,BishopTrainDir,BishopValDir,splitSize)
split_data(KingSourceDir,KingTrainDir,KingValDir,splitSize)
split_data(KnightSourceDir,KnightTrainDir,KnightValDir,splitSize)
split_data(PawnSourceDir,PawnTrainDir,PawnValDir,splitSize)
split_data(QueenSourceDir,QueenTrainDir,QueenValDir,splitSize)
split_data(RookSourceDir,RookTrainDir,RookValDir,splitSize)