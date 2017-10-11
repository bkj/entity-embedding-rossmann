#!/usr/bin/env python

"""
    prepare_features.py
"""

import random
import pickle
import numpy as np
from datetime import datetime
from sklearn import preprocessing

random.seed(42)

def feature_list(record):
    promo = int(record['Promo'])
    store_index = int(record['Store'])
    day_of_week = int(record['DayOfWeek'])
    
    dt = datetime.strptime(record['Date'], '%Y-%m-%d')
    year = dt.year
    month = dt.month
    day = dt.day
    
    state = store_data[store_index - 1]['State']
    
    return [
        promo,
        store_index,
        day_of_week,
        year,
        month,
        day,
        state,
    ]


train_data = pickle.load(open('./data/train_data.pickle', 'rb'))
store_data = pickle.load(open('./data/store_data.pickle', 'rb'))

train_data_X = []
train_data_y = []

for record in train_data:
    if record['Sales'] != '0' and record['Open'] != '':
        train_data_X.append(feature_list(record))
        train_data_y.append(int(record['Sales']))

train_data_X = np.array(train_data_X)
train_data_y = np.array(train_data_y)

label_encoders = []
for i in range(train_data_X.shape[1]):
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(train_data_X[:, i])
    train_data_X[:, i] = label_encoder.transform(train_data_X[:, i])
    label_encoders.append(label_encoder)

train_data_X = train_data_X.astype(int)

pickle.dump(label_encoders, open('./data/label_encoders.pickle', 'wb'), -1)
pickle.dump((train_data_X, train_data_y), open('./data/feature_train_data.pickle', 'wb'), -1)
