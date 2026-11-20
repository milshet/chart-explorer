#!/usr/bin/env bash
# LLM-as-judge on the local GPU (tasks 3.7, 3.13): openai/gpt-oss-20b by default. Options go to scripts/judge.py run.
#
#   cd ~/chart-explorer && mkdir -p logs
#   python scripts/judge.py sample --name run_a_r16                  # once; commit results/judge/run_a_r16/sample.jsonl
#   python scripts/judge.py run --name run_a_r16 --dry-run           # login shell: renders the prompts, no GPU
#   sbatch slurm/judge.sbatch --name run_a_r16
#   sbatch slurm/judge.sbatch --name run_a_r16 --model microsoft/Phi-4-mini-instruct   # fallback, only if gpt-oss fails to load
#   tail -f logs/chartqa-judge-<JOB_ID>.out
#
# Results: results/judge/<name>/judged.jsonl and summary.json (judge model, revision, decoding, commit).
# Runs from ~/venv unchanged: without the MXFP4 kernels transformers loads gpt-oss-20b as bf16 (~42 GB), which fits.
#SBATCH --job-name=chartqa-judge
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

[ $# -gt 0 ] || { echo "usage: sbatch slurm/judge.sbatch --name <eval prefix> [judge.py run options]" >&2; exit 2; }

source "$HOME/.tracka_environment"
source "$HOME/venv/bin/activate"
export HF_HUB_OFFLINE=1          # model is already downloaded; never spend GPU time on the network
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python scripts/judge.py run "$@"
echo JUDGE_OK
