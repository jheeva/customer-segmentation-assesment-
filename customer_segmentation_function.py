# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:41:30 2022

@author: End User
"""

#%%Module
import pandas as pd
import re
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Bidirectional
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json


TOKENIZER_PATH = os.path.join(os.getcwd(), 'tokenizer_data2.json')
DATASET_TRAIN_PATH=os.path.join(os.getcwd(),'train.csv')
DATASET_TEST_PATH=os.path.join(os.getcwd(),'new_customers.csv')


#%% Model Creation

class ModelBuilding():
    
    def lstm_layer(self, num_columns, nb_categories,
                   embedding_output=32, nodes=16, dropout=0.2):
        
        model = Sequential()
        model.add(Embedding(num_columns, embedding_output)) 
        model.add(Bidirectional(LSTM(nodes, return_sequences=True))) # added bidirectional
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation="softmax")) 
        model.summary()
        
        return model

    def simple_lstm_layer(self, num_columns, nb_categories,
                   embedding_output=64, nodes=32, dropout=0.2):
        
        model = Sequential()
        model.add(Embedding(num_columns, embedding_output)) 
        model.add(LSTM(nodes, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation="softmax")) 
        model.summary()
        
        return model
    
  
    
    
    
    
    
    
    
 #%% Model evaluation

class ModelEvaluation():
    
    def evaluation(self, y_true, y_pred):
        print(classification_report(y_true, y_pred)) # classification report
        print(confusion_matrix(y_true, y_pred)) # confusion matrix
        print(accuracy_score(y_true, y_pred))

   #%% model testing
   cs = ModelBuilding()
   model = cs.lstm_layer(5000, 2)
  