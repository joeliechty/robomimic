#!/bin/bash

python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/square/square_feats_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/square/square_feats_H1_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/square/square_feats_H2_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/square/square_feats_Q1_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/square/square_feats_Q2_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/square/square_feats_Q3_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/square/square_feats_Q4_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/lift/lift_feats_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/lift/lift_feats_H1_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/lift/lift_feats_H2_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/lift/lift_feats_Q1_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/lift/lift_feats_Q2_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/lift/lift_feats_Q3_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/lift/lift_feats_Q4_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/can/can_feats_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/can/can_feats_H1_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/can/can_feats_H2_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/can/can_feats_Q1_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/can/can_feats_Q2_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/can/can_feats_Q3_w_cdm.hdf5 --ratio 0.1
python robomimic/scripts/split_train_val.py --dataset /app/robomimic/datasets/can/can_feats_Q4_w_cdm.hdf5 --ratio 0.1