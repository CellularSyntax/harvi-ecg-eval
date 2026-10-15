#!/bin/bash
#SBATCH -p gpu -q a100 --gres=gpu:a100:1 --cpus-per-task=2 --mem=64G -t 2:00:00 -J fh_frz_clin -o ${PROJECT_ROOT}/results/harvi/fairecg/_log_frozen_fairecgv2clin.out
umask 002
srun --container-image=${PROJECT_ROOT}/containers/ecg_audit.sqsh --container-mounts=${PROJECT_ROOT}:${PROJECT_ROOT},${FAIRECG_ROOT}:${FAIRECG_ROOT},${HARVI_V2_ROOT}:${HARVI_V2_ROOT} --no-container-mount-home bash -c "export FE=${FAIRECG_ROOT} PYTHONNOUSERSITE=1 TORCH_EXTENSIONS_DIR=${PROJECT_ROOT}/.torch_ext; for k in 0 1 2 3 4; do python3 -u ${PROJECT_ROOT}/results/harvi/_fairecg_harvi.py --alias fairecgv2clin --ckpt ${PROJECT_ROOT}/models/s5eng_xrecipe_wsd16/fairecg_ep15_full.pt --fold \$k --mode frozen --out ${PROJECT_ROOT}/results/harvi/fairecg || exit 1; done"
