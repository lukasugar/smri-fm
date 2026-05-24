#!/usr/bin/env bash
#SBATCH --job-name=hello_gpu
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=slurms/%j.out
#SBATCH --error=slurms/%j.err

# Usage from the repository root on the login node:
#   mkdir -p slurms
#   sbatch scripts/hello_gpu.sbatch
#
# After the job finishes, inspect:
#   cat /admin/home/luka.secerovic/first.txt
#   tail -n 100 slurms/<job_id>.out
#   tail -n 100 slurms/<job_id>.err

set -euo pipefail

OUTPUT_FILE="/admin/home/luka.secerovic/first.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
else
    GPU_COUNT="0"
fi

{
    echo "Hello World"
    echo "Available GPUs: ${GPU_COUNT}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
    echo "Hostname: $(hostname)"
    echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
} > "${OUTPUT_FILE}"

echo "Wrote ${OUTPUT_FILE}"
