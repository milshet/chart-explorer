#!/usr/bin/env bash
# Task 4.1: verified merge of the final adapter into bf16 base weights. Options go to scripts/serve/merge.py.
#
#   cd ~/chart-explorer && mkdir -p logs
#   quota -s                                                                  # the merge writes ~7 GB
#   python scripts/serve/merge.py --adapter ~/runs/run_a_r16/adapter --dry-run   # login shell first
#   sbatch slurm/merge.sbatch --adapter ~/runs/run_a_r16/adapter
#   tail -f logs/chartqa-merge-<JOB_ID>.out
#
# Writes ~/merged/<run>/ (weights + merge_manifest.json; kept: it is the served model) and
# results/serve/merge_<run>.json. The log ends MERGE_VERIFIED, MERGE_FAILED_ANSWER_CHECK or MERGE_REFUSED.
#SBATCH --job-name=chartqa-merge
#SBATCH --partition=vm3_slurm_partition
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -Eeuo pipefail

[ $# -gt 0 ] || { echo "usage: sbatch slurm/merge.sbatch --adapter <run>/adapter [merge.py options]" >&2; exit 2; }

source "$HOME/.tracka_environment"
source "$HOME/venv/bin/activate"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python scripts/serve/merge.py "$@"
