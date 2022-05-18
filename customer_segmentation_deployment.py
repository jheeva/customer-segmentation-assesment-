# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:16:05 2022

@author: End User
"""

from tensorflow.keras.models import load_model
import os
import json
import numpy as np
from customer_segmentation_function import ExploratoryDataAnalysis
from tensorflow.keras.preprocessing.text import tokenizer_from_json


MODEL_PATH = os.path.join(os.getcwd(), 'train.h5')
JSON_PATH = os.path.join(os.getcwd(), "tokenizer_data2.json")

#%% model loading
df_2_clf= load_model(MODEL_PATH)
df_2_clf.summary()



#%%load the tokenizer

with open(JSON_PATH, 'r') as json_file:
    token = json.load(json_file)
    
#%% Step 1) clean the data

eda = ExploratoryDataAnalysis()
new_df = eda.remove_tags(df_2_clf)


#%% Data preprocessing

loaded_tokenizer = tokenizer_from_json(token)

token_new = loaded_tokenizer.texts_to_sequences(new_df)

