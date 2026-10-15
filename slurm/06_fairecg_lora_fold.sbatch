#!/bin/bash
#SBATCH -p gpu -q h100 --gres=gpu:h100:1 --cpus-per-task=2 --mem=48G -t 3:00:00 -J fh_lora_f0 -o ${PROJECT_ROOT}/results/harvi/fairecg/_log_lora_f0.out
umask 002
srun --container-image=${PROJECT_ROOT}/containers/ecg_audit.sqsh --container-mounts=${PROJECT_ROOT}:${PROJECT_ROOT},${FAIRECG_ROOT}:${FAIRECG_ROOT},${HARVI_V2_ROOT}:${HARVI_V2_ROOT} --no-container-mount-home bash -c "export FE=${FAIRECG_ROOT} PYTHONNOUSERSITE=1 TORCH_EXTENSIONS_DIR=${PROJECT_ROOT}/.torch_ext; python3 -u ${PROJECT_ROOT}/results/harvi/_fairecg_harvi.py --ckpt ${PROJECT_ROOT}/models/s5eng_xrecipe_wsd16/fairecg_ep15_full.pt --fold 0 --mode lora --out ${PROJECT_ROOT}/results/harvi/fairecg"
