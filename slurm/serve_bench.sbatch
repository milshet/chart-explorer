#!/usr/bin/env bash
# Served latency and tokens/sec (tasks 3.8, 3.14): merged bf16, transformers.generate, batch 1. Options go to
# scripts/serve_bench.py.
#
#   cd ~/chart-explorer && mkdir -p logs
#   python scripts/serve_bench.py --adapter ~/runs/run_a_r16/adapter --dry-run      # login shell first
#   sbatch slurm/serve_bench.sbatch --adapter ~/runs/run_a_r16/adapter
#   sbatch slurm/serve_bench.sbatch --adapter ~/runs/ablation_answer_only/adapter
#   tail -f logs/chartqa-serve-bench-<JOB_ID>.out
#
# Writes results/serve/<run>.json. Each job writes a ~7 GB merged checkpoint to ~/merged and deletes it on exit:
# run the two jobs one after the other if the quota is tight.
#SBATCH --job-name=chartqa-serve-bench
#SBATCH --partition=vm3_slurm_partition
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -Eeuo pipefail

[ $# -gt 0 ] || { echo "usage: sbatch slurm/serve_bench.sbatch --adapter <run>/adapter [serve_bench.py options]" >&2; exit 2; }

source "$HOME/.tracka_environment"
source "$HOME/venv/bin/activate"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python scripts/serve_bench.py "$@"
