#!/usr/bin/env bash
# QLoRA training for Qwen2.5-VL-3B on VM03 (Week 2). One config per run, see configs/.
#
#   cd ~/chart-explorer && mkdir -p logs
#   python scripts/train.py configs/run_a.yaml --dry-run          # login shell first: checks config + data
#   sbatch slurm/train.sbatch configs/run_a.yaml                  # full run
#   sbatch slurm/train.sbatch configs/run_a.yaml --max-steps 50 --out ~/runs/pilot   # pilot (#3)
#   sbatch slurm/train.sbatch configs/run_a.yaml --resume         # continue after a time limit
#   tail -f logs/chartqa-train-<JOB_ID>.out
#
# Python stops cleanly and saves a checkpoint 15 minutes before the time limit (exit code 75 = resume it).
# GPU-hours count only while the job runs, so the 4-hour request costs nothing extra on a shorter run.
#SBATCH --job-name=chartqa-train
#SBATCH --partition=vm3_slurm_partition
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -Eeuo pipefail

CONFIG="${1:?usage: sbatch slurm/train.sbatch configs/<run>.yaml [train.py options]}"
shift

source "$HOME/.tracka_environment"
source "$HOME/venv/bin/activate"
export HF_HUB_OFFLINE=1          # model is already downloaded; never spend GPU time on the network
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python scripts/train.py "$CONFIG" --max-minutes 225 "$@"
