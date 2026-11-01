#!/bin/bash
# Run every scripts/slam.py command of one experiment as a Slurm job array (one run per task):
#
#   python tools/experiments.py commands table6 > table6_jobs.txt
#   sbatch --array=1-$(wc -l < table6_jobs.txt) slurm/array.sbatch table6_jobs.txt
#   python tools/make_tables.py table6          # once all tasks have finished
#
# Adjust the resources below and the environment activation to your cluster
# (uncomment --account / --partition if your cluster needs them).
#SBATCH --job-name=vcs-slam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
##SBATCH --account=<account>
##SBATCH --partition=<gpu-partition>

set -euo pipefail
JOBS=${1:?usage: sbatch --array=1-N slurm/array.sbatch JOBS_FILE}
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Activate the environment, e.g.:
# source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate vcs-slam

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS")
echo "[task ${SLURM_ARRAY_TASK_ID}] $CMD"
eval "$CMD"
