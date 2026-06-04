#!/bin/sh
#SBATCH --job-name=lmfi_projection
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
## Default GPU fits 7B models; override on the CLI for small models,
## e.g. --gres=gpu:1g.10gb:1 for a gpt2 canary.
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm-logs/projection-%j.out
#SBATCH --error=results/slurm-logs/projection-%j.out

# Projection experiment (the hub): per-component logit projections.
# Submit from the repo root:
#   mkdir -p results/slurm-logs
#   sbatch --export=ALL,MODELS="gpt2",COMPONENT="both" experiments/02_projection/projection.slurm

set -euo pipefail

REPO=/scratch/drai2/dream-project/lm-failure-by-interference
ENV=/scratch/drai2/dream-project/env
cd "$REPO"

module load gnu10 openmpi || true
module load python/3.10.1-5r || true
source "$ENV/bin/activate"

export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODELS="${MODELS:-gpt2}"
COMPONENT="${COMPONENT:-both}"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-NA}  models=${MODELS}  component=${COMPONENT}"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python experiments/02_projection/run.py --component "${COMPONENT}" --models ${MODELS}

echo "=== DONE projection ==="
