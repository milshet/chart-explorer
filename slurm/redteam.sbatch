#!/usr/bin/env bash
# Task 4.5: the red-team log. Starts the endpoint inside this job, sends the 21 red-team charts (data/redteam/)
# through it, scores each attempt (scripts/serve/redteam.py), then stops the endpoint.
#
#   python scripts/serve/redteam.py --dry-run         # login shell first
#   sbatch slurm/redteam.sbatch
#   tail -f $(ls -t logs/chartqa-redteam-*.out | head -1)
#
# Writes results/redteam/run_a_r16.json and the Markdown log beside it. The endpoint's own log goes to
# logs/serve-<JOB_ID>.log.
#SBATCH --job-name=chartqa-redteam
#SBATCH --partition=vm3_slurm_partition
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -Eeuo pipefail

source "$HOME/.tracka_environment"
source "$HOME/venv/bin/activate"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
PORT=$((18000 + SLURM_JOB_ID % 1000))
python scripts/serve/app.py --port "$PORT" > "logs/serve-${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
python scripts/serve/redteam.py --url "http://127.0.0.1:${PORT}" "$@"
