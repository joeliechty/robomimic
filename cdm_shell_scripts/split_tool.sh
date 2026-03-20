#!/bin/bash

python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/tool/tool_feats.hdf5 --split 2
python3 robomimic/utils/divergence_utils.py --dataset /app/robomimic/datasets/tool/tool_feats.hdf5 --split 4
