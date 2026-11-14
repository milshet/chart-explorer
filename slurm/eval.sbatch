#!/usr/bin/env bash
# Evaluation of the base model or a trained adapter on VM03 (#19). Options go straight to scripts/evaluate.py.
#
#   cd ~/chart-explorer && mkdir -p logs
#   python scripts/evaluate.py --adapter ~/runs/run_a_r16/adapter --split val --dry-run   # login shell first
#   sbatch slurm/eval.sbatch --adapter ~/runs/run_a_r16/adapter --split val
#   sbatch slurm/eval.sbatch --adapter ~/runs/run_a_r16/checkpoints/epoch-1 --split val
#   sbatch slurm/eval.sbatch --split test --targets answer_only --shots 0 --precision 4bit --max-new-tokens 384
#   tail -f logs/chartqa-eval-<JOB_ID>.out
#
# Results: ~/evals/<name>/summary.json and predictions.jsonl (or --out).
#SBATCH --job-name=chartqa-eval
#SBATCH --partition=vm3_slurm_partition
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -Eeuo pipefail

[ $# -gt 0 ] || { echo "usage: sbatch slurm/eval.sbatch --split val|test [evaluate.py options]" >&2; exit 2; }

source "$HOME/.tracka_environment"
source "$HOME/venv/bin/activate"
export HF_HUB_OFFLINE=1          # model is already downloaded; never spend GPU time on the network
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python scripts/evaluate.py "$@"
