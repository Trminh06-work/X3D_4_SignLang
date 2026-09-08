#!/bin/bash
#SBATCH --job-name=x3d_eval
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=Virtual
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Score every checkpoint on the test split. The work lives in src/evaluate.py.
# Decoding the clips is the bottleneck, so this wants CPUs rather than a GPU.
#   sbatch script/evaluate.sh
#   sbatch script/evaluate.sh script/train.conf

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

CONFIG_FILE="${1:-script/train.conf}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    printf 'Config file not found: %s\n' "${CONFIG_FILE}" >&2
    exit 1
fi

source "${CONFIG_FILE}"

mkdir -p logs

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

export CHECKPOINT_DIR RESULTS_DIR BATCH_SIZE

"${PYTHON_BIN}" -m src.evaluate
