# -*- coding: utf-8 -*-
"""
Created on Wed May 18 18:50:48 2022

@author: aceso
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from customer_segment import EDA, Model, Evaluation
from tensorflow.keras.models import load_model

# Constant
DATA_PATH = os.path.join(os.getcwd(), "Data","new_customers.csv")
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'Model','model.h5')

#%% EDA
df_new = pd.read_csv(DATA_PATH)

X_raw = df_new.iloc[:,1:10] # features only

eda = EDA()
# label encoder
X_clean = eda.label_encoder(X_raw)

# fill the NaN
X_imputed = eda.simple_imputer(X_clean)

# scale the features
X_scaled = eda.feature_scaling(X_imputed)

# Load Model
model = load_model(MODEL_SAVE_PATH)

prediction = model.predict(X_scaled)

y_pred = np.argmax(prediction, axis = 1)

# concat in dataframe
new_x = pd.DataFrame(X_scaled)
y_pred = pd.DataFrame(y_pred)

# full completed data
updated_data = pd.concat([new_x, y_pred], axis = 1)

