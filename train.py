#!/usr/bin/env python

"""
    train.py
"""

from __future__ import print_function, division

import sys
import pickle
import argparse
import numpy as np
from collections import OrderedDict

np.random.seed(123)
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Concatenate, Input, Dense, Flatten, Embedding

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-size', type=float, default=0.9)
    parser.add_argument('--train-sample', type=float, default=200000)
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-inflate', type=float, default=1.0)
    
    return parser.parse_args()

# --
# Model definition

class Net(object):
    
    def __init__(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=128, max_inflate=1.0):
        
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.num_dense = 1
        self.layer_dims = OrderedDict([
            
            # Original
            # ('store_index', (1115, 10)),
            # ('day_of_week', (7, 6)),
            # ('year', (3, 2)),
            # ('month', (12, 6)),
            # ('day', (31, 10)),
            # ('state', (12, 6)),
            
            # Simplified
            ('store_index', (1115, 10)),
            ('day_of_week', (7, 10)),
            ('year', (3, 10)),
            ('month', (12, 10)),
            ('day', (31, 10)),
            ('state', (12, 10)),
        ])
        
        self.max_log_y = np.max(np.log(np.concatenate([y_train, y_val]))) * max_inflate
        self._init_model()
        self.fit(X_train, y_train, X_val, y_val)
        
    
    def _split_columns(self, X):
        return [x.reshape(-1, 1) for x in X.T]
    
    
    def _init_model(self):
        
        # Dense layers
        inputs = []
        legs   = []
        for _ in range(self.num_dense):
            dense_inp = Input(shape=(1, ))
            dense_out = Dense(1)(dense_inp)
            inputs.append(dense_inp)
            legs.append(dense_out)
        
        # Categorical layers
        emb_legs = []
        for _, (input_dim, output_dim) in self.layer_dims.items():
            emb_inp = Input(shape=(1, ))
            emb_out = Embedding(input_dim, output_dim)(emb_inp)
            emb_out = Flatten()(emb_out)
            inputs.append(emb_inp)
            legs.append(emb_out)
        
        x = Concatenate()(legs)
        x = Dense(1000, activation='relu')(x)
        x = Dense(500, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=inputs, outputs=x)
        self.model.compile(loss=self.loss, optimizer='adam')
    
    def loss(self, y_true, y_pred):
        """ explicitly optimize the loss metric """
        y_true = K.exp(y_true * self.max_log_y)
        y_pred = K.exp(y_pred * self.max_log_y)
        return K.mean(K.abs((y_true - y_pred) / y_true))
        
    def _to_internal(self, val):
        return np.log(val) / self.max_log_y
        
    def _from_internal(self, val):
        return np.exp(val * self.max_log_y)
        
    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            self._split_columns(X_train), self._to_internal(y_train),
            validation_data=(
                self._split_columns(X_val),
                self._to_internal(y_val)
            ),
            epochs=self.epochs,
            batch_size=self.batch_size,
        )
    
    def predict(self, X):
        X = self._split_columns(X)
        preds = self.model.predict(X, verbose=True, batch_size=512).squeeze()
        return self._from_internal(preds)

# --
# Helpers

def sample(X, y, n):
    num_row = X.shape[0]
    indices = np.random.randint(num_row, size=n)
    return X[indices, :], y[indices]


if __name__ == "__main__":
    args = parse_args()
    
    # --
    # IO
    
    X_train = np.load('./data/X_train.npy')
    y_train = np.load('./data/y_train.npy')
    
    
    # --
    # Train/test split
    
    train_size = int(args.train_size * len(X_train))
    X_val, y_val     = X_train[train_size:], y_train[train_size:]
    X_train, y_train = X_train[:train_size], y_train[:train_size]
    
    if args.train_sample:
        X_train, y_train = sample(X_train, y_train, args.train_sample)
    
    # --
    # Train -- ensembling would give better results
    
    model = Net(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_inflate=args.max_inflate
    )
    
    # --
    # Evaluate
    
    train_preds = model.predict(X_train)
    err_train = np.abs((y_train - train_preds) / y_train).mean()
    
    val_preds = model.predict(X_val)
    err_val = np.abs((y_val - val_preds) / y_val).mean()

    print("err_train=%f | err_val=%f" % (err_train, err_val))
