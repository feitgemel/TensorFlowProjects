import numpy as np

# load the data 

print("Start load the data ")
features = np.load("e:/temp/5-vehicales-features.npy")
y_train = np.load("e:/temp//5-vehicales-y_train.npy")
print("Finish load the data")

print(features.shape)
print(y_train.shape)


# XGboost
import xgboost as xgb
model = xgb.XGBClassifier()

print(" Run the train(!!!!) features in the XGBoost model : ")
model.fit(features, y_train)

print("Finish the train of XGBoost with VGG features ")

model.save_model("e:/temp/5-vehicles-XGboost.h5")

print("Finish save the XGBoost model ")


