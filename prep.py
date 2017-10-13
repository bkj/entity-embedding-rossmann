#!/usr/bin/env python

"""
    prep.py
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing

# Load train data
train_data = pd.read_csv(paths['train'], dtype=str)
train_data.columns = map(lambda x: x.lower(), train_data.columns)
train_data = train_data.iloc[::-1].reset_index(drop=True)

# Add state information
state_data = pd.read_csv(paths['state'], dtype=str)
state_data.columns = map(lambda x: x.lower(), state_data.columns)
train_data = pd.merge(train_data, state_data, on='store', how='left', sort=False)

# Typing + variable creation
train_data.open = train_data.open.astype(int)
train_data.sales = train_data.sales.astype(int)
train_data.promo = train_data.promo.astype(int)
train_data.store = train_data.store.astype(int)
train_data.dayofweek = train_data.dayofweek.astype(int)
train_data.date = pd.to_datetime(train_data.date)
train_data['year'] = train_data.date.dt.year
train_data['month'] = train_data.date.dt.month
train_data['day'] = train_data.date.dt.day

# Drop days w/o sales
train_data = train_data[(train_data.sales != 0) & (train_data.open == 1)]

sub = train_data[['promo', 'store', 'dayofweek', 'year', 'month', 'day', 'state']]

for c in sub.columns:
    label_encoder = preprocessing.LabelEncoder()
    _ = label_encoder.fit(sub[c])
    sub[c] = label_encoder.transform(sub[c])

X_train = np.array(sub)
y_train = np.array(train_data.sales)

np.save('./data/X_train', X_train)
np.save('./data/y_train', y_train)
