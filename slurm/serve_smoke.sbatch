#!/usr/bin/env bash
# Serving-stack smoke test (task 3.0f). docs/SERVING.md has the procedure and the decision record.
#
# Decided 2026-11-19: transformers.generate, run from ~/venv (the locked training stack; nothing installed).
# vLLM never goes into ~/venv: it would install its own torch over the measured stack.
#
#   M=~/merged/run_a_r16_smoke                                               # ~7 GB: mind the 35 GB quota
#   SERVE_VENV=$HOME/venv sbatch slurm/serve_smoke.sbatch --merge ~/runs/run_a_r16/adapter --merged $M --engine transformers
#   SERVE_VENV=$HOME/venv sbatch slurm/serve_smoke.sbatch --model $M --engine transformers
#   tail -f logs/chartqa-serve-smoke-<JOB_ID>.out
#
# Without SERVE_VENV the job activates ~/venv-serve, for a separate vLLM venv should one ever fit.
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
source "${SERVE_VENV:-$HOME/venv-serve}/bin/activate"   # SERVE_VENV=$HOME/venv for the transformers fallback
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python scripts/serve_smoke.py --log "logs/serve_smoke-${SLURM_JOB_ID}.json" "$@"
