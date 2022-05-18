# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:47:51 2022

@author: End User
"""

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
import datetime

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from customer_segmentation_function import  ModelBuilding 
from customer_segmentation_function import ModelEvaluation



#%%

TOKENIZER_PATH = os.path.join(os.getcwd(), 'tokenizer_data2.json')
PATH_LOGS = os.path.join(os.getcwd(), 'log')
log_dir = os.path.join(PATH_LOGS, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'train.h5')

#%%
DATASET_TRAIN_PATH=os.path.join(os.getcwd(),'train.csv')
DATASET_TEST_PATH=os.path.join(os.getcwd(),'new_customers.csv')



df_1=pd.read_csv(DATASET_TRAIN_PATH)
df_2=pd.read_csv(DATASET_TEST_PATH)

df_1.info()
df_1.describe().T


#x_train=X_train['Open']
#x_test=X_test['Open'].values


#steP3)data cleaning

enc=LabelEncoder()
df_1['Gender']=enc.fit_transform(df_1['Gender'])
df_1['Ever_Married']=enc.fit_transform(df_1['Ever_Married'])
df_1['Profession']=enc.fit_transform(df_1['Profession'])
df_1['Spending_Score']=enc.fit_transform(df_1['Spending_Score'])
df_1['Var_1']=enc.fit_transform(df_1['Var_1'])
df_1['Segmentation']=enc.fit_transform(df_1['Segmentation'])
df_1.info()

X_train=[]
Y_train=[]

X_test=[]
Y_test=[]


# Train test split(X = review, y = sentiment)
X_train, X_test, y_train, y_test = train_test_split(df_1, 
                                                   df_2, 
                                                    test_size=0.3, 
                                                    random_state=125)


# expand training data into 3D array
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

#%% Model Building

mb = ModelBuilding()
num_columns =12
nb_categories = len(df_2.unique())

model = mb.lstm_layer(num_columns, nb_categories)
model.compile(optimizer="adam", 
              loss="categorical_crossentropy", 
              metrics="acc")

tensorboard = TensorBoard(log_dir, histogram_freq=1)

model.fit(X_train, y_train, epochs=3, 
          validation_data=(X_test, y_test),
          callbacks=tensorboard)


#%% Model Evaluation
predicted_advanced = np.empty([len(X_test), 2]) 
for i, test in enumerate(X_test):
    predicted_advanced[i,:] = model.predict(np.expand_dims(test, axis=0))

# Model analysis
y_pred = np.argmax(predicted_advanced, axis=1) 
y_true = np.argmax(y_test, axis=1)

evals = ModelEvaluation()
result = evals.evaluation(y_true, y_pred)


#%% Model Saving
model.save(MODEL_SAVE_PATH)









