#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
cd ${PBS_O_WORKDIR}
source activate ENV_NAME
python main_GAN.py