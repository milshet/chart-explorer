#!/usr/bin/env bash
# Task 4.2: quantise the verified merge (language tower, 4-bit), inspect the export, score bf16 and quantised on test,
# accept or reject. Options go to scripts/serve/quantise.py.
#
#   python scripts/serve/quantise.py --dry-run        # in ~/venv-quant, on the login shell first
#   quota -s                                           # the export adds ~3 GB
#   sbatch slurm/quantise.sbatch
#   tail -f logs/chartqa-quantise-<JOB_ID>.out
#
# Runs in ~/venv-quant: ~/venv's locked stack (via .pth) plus the quantisation backend, installed with
# -c requirements-vm03.lock so no locked package moves. Never install the backend into ~/venv.
#SBATCH --job-name=chartqa-quantise
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

source "$HOME/.tracka_environment"
source "${QUANT_VENV:-$HOME/venv-quant}/bin/activate"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python scripts/serve/quantise.py "$@"
