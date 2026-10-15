#!/bin/bash
#SBATCH -J harvi_oof205
#SBATCH -p gpu
#SBATCH -q h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0:30:00
#SBATCH -o ${PROJECT_ROOT}/results/harvi/_oof_harvi_xlstm205.out
umask 002
srun --container-image='nvcr.io#nvidia/pytorch:25.08-py3' --container-mounts=${PROJECT_ROOT}:${PROJECT_ROOT},${HARVI_V2_ROOT}:${HARVI_V2_ROOT} --container-workdir=${PROJECT_ROOT}/results/harvi/code_snapshot --no-container-mount-home bash -lc '
set -euo pipefail
export PYTHONNOUSERSITE=1 WANDB_MODE=disabled TORCH_EXTENSIONS_DIR=${PROJECT_ROOT}/results/harvi/.torch_ext PIP_CACHE_DIR=${PROJECT_ROOT}/results/harvi/.pipcache
pip install -q -r requirements.txt "xlstm==2.0.5" && python -c "import importlib.metadata as m; print(\"XLSTM\", m.version(\"xlstm\"))"
python evaluate_oof_lightning.py --config configs/eval_oof_default.yaml --config configs/targets/ntprobnp.yaml \
  --data.processed_data ${HARVI_V2_ROOT}/cache/ntprobnp --data.num_workers 7 \
  --eval.ckpt_dir ${PROJECT_ROOT}/results/harvi/ckpts --eval.output_dir ${PROJECT_ROOT}/results/harvi/oof_harvi_xlstm205 --eval.log_plots false
'
