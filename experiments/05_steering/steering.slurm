#!/bin/sh
#SBATCH --job-name=lmfi_steering
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
## Default GPU fits 7B models; override on the CLI for small models,
## e.g. --gres=gpu:1g.10gb:1 for a gpt2 canary.
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=2-00:00:00
#SBATCH --output=results/slurm-logs/steering-%j.out
#SBATCH --error=results/slurm-logs/steering-%j.out

# Unified steering (RASteer). Submit from the repo root, e.g.:
#   sbatch --export=ALL,COMPONENT=attn,MODE=sweep,SPLIT=dev,MODELS="gpt2" \
#          experiments/05_steering/steering.slurm
# Then analyze (CPU is fine but this works too) and apply:
#   sbatch --export=ALL,COMPONENT=attn,MODE=analyze experiments/05_steering/steering.slurm
#   sbatch --export=ALL,COMPONENT=attn,MODE=apply,MODELS="gpt2" experiments/05_steering/steering.slurm

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

COMPONENT="${COMPONENT:-attn}"
MODE="${MODE:-sweep}"
SPLIT="${SPLIT:-dev}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} component=$COMPONENT mode=$MODE split=$SPLIT models=${MODELS:-<config>}"
nvidia-smi -L || true

ARGS="--component $COMPONENT --mode $MODE --split $SPLIT"
if [ -n "${MODELS:-}" ]; then ARGS="$ARGS --models $MODELS"; fi
if [ -n "${METRIC:-}" ]; then ARGS="$ARGS --metric $METRIC"; fi

python experiments/05_steering/run.py $ARGS
echo "=== DONE steering ==="
