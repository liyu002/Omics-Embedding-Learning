#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
cd ${PBS_O_WORKDIR}
source activate ENV_NAME
python train_test.py --omics_mode c --data_root ../dataset/test --file_format csv --seed 111