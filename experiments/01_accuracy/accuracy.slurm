#!/bin/sh
#SBATCH --job-name=lmfi_accuracy
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
## Default GPU fits 7B models; override on the CLI for small models,
## e.g. --gres=gpu:1g.10gb:1 for the gpt2 canary (a 10GB MIG slice).
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm-logs/accuracy-%j.out
#SBATCH --error=results/slurm-logs/accuracy-%j.out

# Accuracy experiment (paper Table 3 / Appendix A), migration canary.
# Submit from the repo root:
#   mkdir -p results/slurm-logs
#   sbatch --export=ALL,MODELS="gpt2" experiments/01_accuracy/accuracy.slurm
# MODELS (space-separated) overrides the model set; defaults to the canary gpt2.

set -euo pipefail

REPO=/scratch/drai2/dream-project/lm-failure-by-interference
ENV=/scratch/drai2/dream-project/env

cd "$REPO"

# Cluster mechanics mirror the project's known-good setup.
module load gnu10 openmpi || true
module load python/3.10.1-5r || true
source "$ENV/bin/activate"

# src-layout: make `import lmfi` work without an editable install.
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
# Weights are already in the HF hub cache; stay offline so a compute node with
# no internet does not hang trying to reach the hub.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODELS="${MODELS:-gpt2}"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-NA}  models=${MODELS}"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python experiments/01_accuracy/run.py --models ${MODELS}

echo "=== DONE accuracy ==="
