#!/bin/bash

python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/lift/low_dim_v15.hdf5
python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/lift/low_dim_v15_H1.hdf5
python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/lift/low_dim_v15_H2.hdf5
python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/lift/low_dim_v15_Q1.hdf5
python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/lift/low_dim_v15_Q2.hdf5
python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/lift/low_dim_v15_Q3.hdf5
python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/lift/low_dim_v15_Q4.hdf5

