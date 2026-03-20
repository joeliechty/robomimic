#!/bin/bash

python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/tool/tool_feats.hdf5
python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/tool/tool_feats_H1.hdf5
python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/tool/tool_feats_H2.hdf5
python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/tool/tool_feats_Q1.hdf5
python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/tool/tool_feats_Q2.hdf5
python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/tool/tool_feats_Q3.hdf5
python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/tool/tool_feats_Q4.hdf5

