#!/bin/bash
#SBATCH -J harvi_ext
#SBATCH -p gpu
#SBATCH -q h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -o ${PROJECT_ROOT}/results/harvi/external/_harvi_ext.out
umask 002
srun --container-image='nvcr.io#nvidia/pytorch:25.08-py3' --container-mounts=${PROJECT_ROOT}:${PROJECT_ROOT},${HARVI_V2_ROOT}:${HARVI_V2_ROOT} --container-workdir=${PROJECT_ROOT}/results/harvi/code_snapshot --no-container-mount-home bash -lc '
set -euo pipefail
export PYTHONNOUSERSITE=1 WANDB_MODE=disabled TORCH_EXTENSIONS_DIR=${PROJECT_ROOT}/results/harvi/.torch_ext PIP_CACHE_DIR=${PROJECT_ROOT}/results/harvi/.pipcache
pip install -q -r requirements.txt "xlstm==2.0.5"
python -u ${PROJECT_ROOT}/results/harvi/external/_harvi_external.py ${PROJECT_ROOT}/results/harvi/code_snapshot ${PROJECT_ROOT}/results/harvi ${HARVI_V2_ROOT}/cache/ntprobnp
'
