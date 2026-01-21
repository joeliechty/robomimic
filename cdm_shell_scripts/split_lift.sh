#!/bin/bash

python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/lift/low_dim_v15.hdf5 --split 2
python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/lift/low_dim_v15.hdf5 --split 4
