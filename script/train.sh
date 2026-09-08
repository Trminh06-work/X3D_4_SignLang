#!/bin/bash
#SBATCH --job-name=x3d_autsl
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=g48-2gpu-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH -o logs/%x_l_%a.out
#SBATCH -e logs/%x_l_%a.err

# Train one X3D variant per array task. The work itself lives in src/main.py.
# The --array range above covers every entry in JOBS; override it for a subset.
#   sbatch script/train.sh
#   sbatch --array=0-5 script/train.sh
#   sbatch script/train.sh script/train.conf

set -eo pipefail

# Slurm runs the batch script from a spool copy, so take the root from the
# directory sbatch was called in. Submit from the project root.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

CONFIG_FILE="${1:-script/train.conf}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    printf 'Config file not found: %s\n' "${CONFIG_FILE}" >&2
    exit 1
fi

source "${CONFIG_FILE}"

mkdir -p logs

# This task's entry from JOBS, split into size and number of frozen blocks
IFS=',' read -r -a JOB_LIST <<< "${JOBS}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID >= ${#JOB_LIST[@]} )); then
    printf 'Task %s is out of range for %s jobs\n' "${TASK_ID}" "${#JOB_LIST[@]}" >&2
    exit 1
fi

IFS=':' read -r SIZE FROZEN <<< "${JOB_LIST[TASK_ID]}"

# Decode the clips once, so that the epochs read frames rather than mp4s. The
# default sits beside the data, which is known to have room for it; /tmp on this
# cluster does not. One array task builds it and the rest wait on its lock.
CACHE_DIR="${CACHE_DIR:-cache}"

mkdir -p "${CACHE_DIR}"

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

export SIZE FROZEN SUBSET EPOCHS BATCH_SIZE LEARNING_RATE CACHE_DIR CHECKPOINT_DIR FIGURE_DIR

"${PYTHON_BIN}" -m src.main
