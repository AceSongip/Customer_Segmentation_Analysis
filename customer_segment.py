# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:50:54 2022

@author: aceso
"""

#%% Modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#%% Constant
DATA_PATH = os.path.join(os.getcwd(), "Data","train.csv")


#%% EDA
class EDA():
    
    def __init__(self):
        pass
    
    def label_encoder(self,df):
        encoder = LabelEncoder()
        df["Gender"] = encoder.fit_transform(df["Gender"])
        df["Ever_Married"] = encoder.fit_transform(df["Ever_Married"])
        df["Graduated"] = encoder.fit_transform(df["Graduated"])
        df["Profession"] = encoder.fit_transform(df["Profession"])
        df["Spending_Score"] = encoder.fit_transform(df["Spending_Score"])
        df["Var_1"] = encoder.fit_transform(df["Var_1"])
        
        return df
        
    def simple_imputer(self,X):
        imputer = SimpleImputer(strategy="median")
        
        return imputer.fit_transform(X)
    
    def feature_scaling(self, X):
        scaler = MinMaxScaler()
        
        return scaler.fit_transform(X)
    
    def one_hot_encoding(self, df):
        encoder = OneHotEncoder(sparse=False)
        
        return encoder.fit_transform(df)
        
class Model():
    
    def neural_network(self, nb_features, nodes1, nodes2, activation1, 
                       nb_target, out_activation):
        
        seq = Sequential(name=("Customer_Segmentation"))
        
        seq.add(Input(shape=(nb_features), name="input_layer"))
        seq.add(Dense(nodes1, activation=activation1, name="layer_1"))
        seq.add(Dropout(0.2))
        seq.add(Dense(nodes2, activation=activation1, name="layer_2"))
        seq.add(Dense(nodes1, activation=activation1, name="layer_3"))
        seq.add(Dense(nb_target, activation=out_activation, name="output_layer"))
        print(seq.summary())
        
        return seq
    
class Evaluation():
    
    def model_eval(self, y_true, y_pred):
    
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred) # argx max find highest value only, 
        cr = classification_report(y_true, y_pred)
        print(cr)
    
    
#%% Testing

if __name__ == "__main__":
    
    df_raw = pd.read_csv(DATA_PATH)
    X_raw = df_raw.iloc[:,1:10] # features only
    
    eda = EDA()
    # label encoder
    X_clean = eda.label_encoder(X_raw)
    
    # fill the NaN
    X_imputed = eda.simple_imputer(X_clean)
    
    # scale the features
    X_scaled = eda.feature_scaling(X_imputed)
    
    # convert target to one_hot encoding
    y_raw = np.expand_dims(df_raw["Segmentation"], axis=-1)
    y_encoded = eda.one_hot_encoding(y_raw)
    
    # assign the features and target
    X = X_scaled
    y = y_encoded
    
    # split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Instantiate model
    model = Model()
    # fit the data
    clf = model.neural_network(nb_features=9, nodes1=64, nodes2=86
                           ,activation1="relu", nb_target=4, 
                           out_activation="softmax")



