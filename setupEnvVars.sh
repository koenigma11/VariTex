#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
export DP=/home/matthias/ETH/Thesis/VariTexLocal/datasets
export FP=/home/matthias/ETH/Thesis/VariTexLocal/datasets
export OP=/home/matthias/ETH/Thesis/VariTexLocal/VariTex/output
export CP=/home/matthias/ETH/Thesis/VariTexLocal/datasets/pretrained/ep44.ckpt


CUDA_VISIBLE_DEVICES=0 python varitex/train.py


export BP=/home/matthias/ETH/Thesis/VariTexLocal
if ! [[ $PYTHONPATH =~ $BP/VariTex ]]
then
  export PYTHONPATH=$PP/VariTex/:$PYTHONPATH;
fi
export DP=$BP/datasets
export FP=$BP/datasets
export OP=$BP/VariTex/output
export CP=$BP/datasets/pretrained/ep44.ckpt