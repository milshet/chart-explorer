#!/usr/bin/env bash
# Task 4.3: run the endpoint (FastAPI + transformers.generate) on the verified merge. Options go to scripts/serve/app.py.
#
#   sbatch slurm/serve.sbatch                                   # serves ~/merged/run_a_r16 on 127.0.0.1:8000
#   grep -m1 Uvicorn logs/chartqa-serve-<JOB_ID>.out             # "Uvicorn running on ..." once the model is loaded
#   curl -s localhost:8000/health                                # on VM03
#   ssh -N -L 8000:127.0.0.1:8000 -p 1026 learner01@<vm03>      # from a laptop, then http://localhost:8000/docs
#   scancel <JOB_ID>                                             # stop it: it holds the GPU while it runs
#
# The app refuses a checkpoint whose merge_manifest.json is not VERIFIED (task 4.1).
#SBATCH --job-name=chartqa-serve
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

source "$HOME/.tracka_environment"
source "$HOME/venv/bin/activate"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
exec python scripts/serve/app.py "$@"
