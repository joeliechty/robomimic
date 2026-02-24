#!/bin/bash

python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/lift/lift_feats.hdf5 --split 2
python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/lift/lift_feats.hdf5 --split 4
