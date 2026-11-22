#!/usr/bin/env bash
# Task 4.3: served latency through HTTP. Starts the endpoint inside this job, replays the 50 requests task 3.14 timed
# in-process (scripts/serve/http_bench.py), then stops the endpoint.
#
#   python scripts/serve/http_bench.py --dry-run     # login shell first
#   sbatch slurm/serve_http_bench.sbatch
#   tail -f logs/chartqa-http-bench-<JOB_ID>.out
#
# Writes results/serve/run_a_r16_http.json. The endpoint's own log goes to logs/serve-<JOB_ID>.log.
#SBATCH --job-name=chartqa-http-bench
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
python scripts/serve/http_bench.py --url "http://127.0.0.1:${PORT}" "$@"
