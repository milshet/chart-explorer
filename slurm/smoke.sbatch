#!/usr/bin/env bash
# Smoke test: Qwen2.5-VL-3B QLoRA pipeline on VM03 (about 10 minutes of GPU time).
#
#   cd ~/chart-explorer && mkdir -p logs
#   sbatch slurm/smoke.sbatch
#   squeue --me
#   tail -n 40 logs/chartqa-smoke-<JOB_ID>.out
#
#SBATCH --job-name=chartqa-smoke
#SBATCH --partition=vm3_slurm_partition
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -Eeuo pipefail

source "$HOME/.tracka_environment"
source "$HOME/venv/bin/activate"
export HF_HUB_OFFLINE=1          # model is already downloaded; never spend GPU time on the network
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python scripts/smoke_test.py \
  --data-dir data/v2.0 \
  --variant image_only \
  --out "$HOME/runs/smoke-$SLURM_JOB_ID"
