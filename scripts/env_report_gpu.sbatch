#!/usr/bin/env bash
#SBATCH --job-name=env_report_gpu
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
#   sbatch scripts/env_report_gpu.sbatch
#
# After the job finishes, inspect:
#   cat /admin/home/luka.secerovic/second.txt
#   tail -n 100 slurms/<job_id>.out
#   tail -n 100 slurms/<job_id>.err

set -euo pipefail

OUTPUT_FILE="/admin/home/luka.secerovic/second.txt"

{
    echo "Environment report"
    echo "=================="
    echo "Hostname: $(hostname)"
    echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
    echo

    echo "uv availability"
    echo "---------------"
    if command -v uv >/dev/null 2>&1; then
        echo "uv available: yes"
        echo "uv path: $(command -v uv)"
        echo "uv version: $(uv --version)"
    else
        echo "uv available: no"
    fi
    echo

    echo "Python and pip"
    echo "--------------"
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    else
        PYTHON_CMD=""
    fi

    if [[ -n "${PYTHON_CMD}" ]]; then
        echo "python path: $(command -v "${PYTHON_CMD}")"
        echo "python version: $("${PYTHON_CMD}" --version 2>&1)"
        echo
        echo "pip list"
        echo "--------"
        "${PYTHON_CMD}" -m pip list
    elif command -v pip >/dev/null 2>&1; then
        echo "python path: unavailable"
        echo "pip path: $(command -v pip)"
        echo
        echo "pip list"
        echo "--------"
        pip list
    else
        echo "python path: unavailable"
        echo "pip path: unavailable"
        echo "pip list unavailable: no python or pip command found"
    fi
} > "${OUTPUT_FILE}" 2>&1

echo "Wrote ${OUTPUT_FILE}"
