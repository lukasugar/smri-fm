#!/usr/bin/env bash
#SBATCH --job-name=filesystem_report_gpu
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
#   sbatch scripts/filesystem_report_gpu.sbatch
#
# After the job finishes, inspect:
#   cat /admin/home/luka.secerovic/third.txt
#   tail -n 100 slurms/<job_id>.out
#   tail -n 100 slurms/<job_id>.err

set -euo pipefail

# Output to login node
# OUTPUT_FILE="/admin/home/luka.secerovic/third.txt"

# Output to the shared data folder (much bigger storage)
mkdir -p /data/luka.secerovic
OUTPUT_FILE="/data/luka.secerovic/third.txt"

section() {
    echo
    echo "$1"
    printf '%*s\n' "${#1}" '' | tr ' ' '-'
}

run_or_note() {
    local label="$1"
    shift

    section "${label}"
    if "$@"; then
        true
    else
        echo "Command failed: $*"
    fi
}

{
    echo "Filesystem report"
    echo "================="
    echo "Hostname: $(hostname)"
    echo "Date: $(date --iso-8601=seconds 2>/dev/null || date)"
    echo "User: $(id 2>/dev/null || whoami)"
    echo "PWD: $(pwd)"
    echo "HOME: ${HOME:-unset}"
    echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
    echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-unset}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

    run_or_note "PATH" printenv PATH
    run_or_note "df -hT" df -hT
    run_or_note "findmnt" findmnt
    run_or_note "/proc/mounts" cat /proc/mounts

    section "Important Directory Visibility"
    for path in \
        /admin \
        /admin/home \
        /admin/home/luka.secerovic \
        /data \
        /tmp \
        /scratch \
        /mnt \
        /opt \
        /usr/local \
        "$HOME" \
        "$PWD"; do
        if [[ -e "$path" ]]; then
            echo "[exists] $path"
            ls -ld "$path" 2>&1 || true
        else
            echo "[missing] $path"
        fi
    done

    section "Top-Level Root Listing"
    ls -la /

    section "Home Listing"
    ls -la "$HOME"

    section "Data Listing"
    if [[ -d /data ]]; then
        ls -la /data
    else
        echo "/data is not visible"
    fi

    section "Writable Checks"
    for path in "$HOME" /data /tmp; do
        if [[ -d "$path" && -w "$path" ]]; then
            echo "[writable] $path"
        elif [[ -d "$path" ]]; then
            echo "[not writable] $path"
        else
            echo "[missing] $path"
        fi
    done
} > "${OUTPUT_FILE}" 2>&1

echo "Wrote ${OUTPUT_FILE}"
