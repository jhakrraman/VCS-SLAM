#!/bin/bash
# One SLAM run; all arguments are passed to scripts/slam.py:
#
#   sbatch slurm/run.sbatch configs/replica/slam.py --scene room0
#   sbatch slurm/run.sbatch configs/replica/slam.py --scene room0 --tag no_vcsu --set semantics.vcsu=false
#
# Adjust the resources below and the environment activation to your cluster
# (uncomment --account / --partition if your cluster needs them).
#SBATCH --job-name=vcs-slam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
##SBATCH --account=<account>
##SBATCH --partition=<gpu-partition>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Activate the environment, e.g.:
# source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate vcs-slam

python scripts/slam.py "$@" --skip-existing
