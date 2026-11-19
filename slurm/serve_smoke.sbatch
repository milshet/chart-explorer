#!/usr/bin/env bash
# Serving-stack smoke test (task 3.0f), in ~/venv-serve -- never ~/venv, whose torch the measured runs depend on.
# docs/SERVING.md has the whole procedure (create the venv, resolve, freeze requirements-serve.lock).
#
#   sbatch slurm/serve_smoke.sbatch --merge ~/runs/run_a_r16/adapter --merged ~/merged/run_a_r16_smoke
#   sbatch slurm/serve_smoke.sbatch --model ~/merged/run_a_r16_smoke                    # vLLM
#   sbatch slurm/serve_smoke.sbatch --model ~/merged/run_a_r16_smoke --engine transformers  # the fallback
#   tail -f logs/chartqa-serve-smoke-<JOB_ID>.out
#SBATCH --job-name=chartqa-serve-smoke
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
source "$HOME/venv-serve/bin/activate"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python scripts/serve_smoke.py --log "logs/serve_smoke-${SLURM_JOB_ID}.json" "$@"
