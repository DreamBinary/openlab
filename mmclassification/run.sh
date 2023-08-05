#!/bin/bash
module load anaconda/2022.10
module load cuda/11.1
module load gcc/7.3
source activate ai
export PYTHONUNBUFFERED=1
python tools/train.py configs/cifar10/cifar10_config.py
