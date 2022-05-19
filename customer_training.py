# -*- coding: utf-8 -*-
"""
Created on Wed May 18 18:03:35 2022

@author: aceso
"""

#%%modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from sklearn.model_selection import train_test_split
from customer_segment import EDA, Model, Evaluation
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import plot_model
import pickle
import seaborn as sns

# Constant
DATA_PATH = os.path.join(os.getcwd(), "Data","train.csv")
PATH_LOGS = os.path.join(os.getcwd(), 'Log')
log_dir = os.path.join(PATH_LOGS, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'Model','model.h5')

#%%EDA

df_raw = pd.read_csv(DATA_PATH)

#%% Data Inspection using plotting
# Gender against segmentation (A,B,C,D)
sns.countplot(df_raw["Gender"], hue=df_raw["Segmentation"]) # Most male categorize under
# D catergory and female and male do not have obvious differences in trend

# profesion vs segmentation
sns.countplot(df_raw["Profession"], hue=df_raw["Segmentation"]) # Only artist and
# Healthcare show trend, others weak trend

# ever_married vs segmentation
sns.countplot(df_raw["Ever_Married"], hue=df_raw["Segmentation"]) # there's strong trend

# [Altenative] How to compare groupby more than 2 cat and plot it?
df_raw.groupby(["Segmentation", "Ever_Married", "Gender"]).agg({"Segmentation":"count"}).plot(kind="bar")

#%% Data Cleaning
X_raw = df_raw.iloc[:,1:10] # features only

eda = EDA()
# label encoder
X_clean = eda.label_encoder(X_raw)

# fill the NaN
X_imputed = eda.simple_imputer(X_clean)

#%% Feature Scalling
# scale the features
X_scaled = eda.feature_scaling(X_imputed)

#%% Data Preprocessing
# convert target to one_hot encoding
y_raw = np.expand_dims(df_raw["Segmentation"], axis=-1)
y_encoded = eda.one_hot_encoding(y_raw)

# assign the features and target
X = X_scaled
y = y_encoded

# split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Instantiate model
model = Model()
# fit the data
clf = model.neural_network(nb_features=9, nodes1=64, nodes2=86,
                       activation1="relu", nb_target=4, 
                       out_activation="softmax")
clf.compile(optimizer="adam",
            loss="categorical_crossentropy",
            metrics="acc")

# plot the model
plot_model(clf)

# create earlystopping
es = EarlyStopping(monitor="val_loss", patience=5)
# create tensorboard
tensorboard = TensorBoard(log_dir, histogram_freq=1)

nn = clf.fit(X_train, y_train, epochs=100, validation_data=(X_test,y_test),
             callbacks=[es, tensorboard])

result = clf.predict(X_test)

# model saving
clf.save(MODEL_SAVE_PATH)


#%% Model evaluation (HAVING PROBLEM TO PREDICT)

pred_x = clf.predict(X)
y_true = np.argmax(y, axis =1)
y_pred = np.argmax(pred_x, axis = 1)

eval = Evaluation()
result = eval.model_eval(y_true, y_pred)






