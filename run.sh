#!/bin/bash

# run.sh

python extract_csv_files.py
python prepare_features.py
python train.py