#!/usr/bin/env bash
# Task 4.8: smoke-test the demo in the order it is given. Starts a fresh endpoint inside this job, makes the demo's
# requests (scripts/serve/demo.py STEPS) one after another, records what came back, then stops the endpoint.
#
#   sbatch slurm/demo_smoke.sbatch
#   tail -f $(ls -t logs/chartqa-demo-smoke-*.out | head -1)
#
# Writes results/demo/smoke.json; then, on any machine: python scripts/serve/demo.py --write
#SBATCH --job-name=chartqa-demo-smoke
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
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
PORT=$((18000 + SLURM_JOB_ID % 1000))
python scripts/serve/app.py --port "$PORT" > "logs/serve-${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
python scripts/serve/demo.py --smoke --url "http://127.0.0.1:${PORT}" "$@"
