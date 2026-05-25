#!/usr/bin/env bash
#SBATCH --job-name=s3_report_gpu
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
#   sbatch scripts/s3_report_gpu.sbatch s3://medarc/luka.secerovic/test-output
#
# The script writes a file named fourth.txt under the S3 prefix you pass:
#   s3://medarc/luka.secerovic/test-output/fourth.txt
#
# For Cloudflare R2, export these before submitting, or define them in a private
# shell file and source it before sbatch. Do not commit credentials to git.
#   export AWS_ENDPOINT_URL="https://<account-id>.r2.cloudflarestorage.com"
#   export AWS_ACCESS_KEY_ID="..."
#   export AWS_SECRET_ACCESS_KEY="..."
#   export AWS_REGION="auto"
#   sbatch scripts/s3_report_gpu.sbatch s3://medarc/luka.secerovic/test-output
#
# Alternative invocation using an environment variable:
#   export S3_OUTPUT_URI=s3://medarc/luka.secerovic/test-output
#   sbatch scripts/s3_report_gpu.sbatch
#
# After the job finishes, inspect:
#   aws s3 cp s3://medarc/luka.secerovic/test-output/fourth.txt -
#   tail -n 100 slurms/<job_id>.out
#   tail -n 100 slurms/<job_id>.err

set -euo pipefail

DESTINATION="${1:-${S3_OUTPUT_URI:-}}"

if [[ -z "${DESTINATION}" ]]; then
    echo "Usage: sbatch scripts/s3_report_gpu.sbatch s3://bucket/prefix" >&2
    echo "Or set S3_OUTPUT_URI=s3://bucket/prefix before submitting." >&2
    exit 2
fi

if [[ "${DESTINATION}" != s3://* ]]; then
    echo "Destination must start with s3://, got: ${DESTINATION}" >&2
    exit 2
fi

if ! command -v aws >/dev/null 2>&1; then
    echo "aws CLI not found in PATH. Install/load aws-cli before running this job." >&2
    exit 1
fi

REMOTE_URI="${DESTINATION%/}/fourth.txt"
LOCAL_FILE="${TMPDIR:-/tmp}/fourth-${SLURM_JOB_ID:-manual}.txt"
DOWNLOADED_FILE="${TMPDIR:-/tmp}/fourth-${SLURM_JOB_ID:-manual}-downloaded.txt"

write_report() {
    cat > "${LOCAL_FILE}" <<EOF
S3 report
=========
Hostname: $(hostname)
Date: $(date --iso-8601=seconds 2>/dev/null || date)
User: $(id 2>/dev/null || whoami)
PWD: $(pwd)
HOME: ${HOME:-unset}
SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}
SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-unset}
CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}

Destination prefix: ${DESTINATION}
Remote file: ${REMOTE_URI}

AWS CLI
-------
aws path: $(command -v aws)
aws version: $(aws --version 2>&1)

AWS environment
---------------
AWS_ENDPOINT_URL set: $([[ -n "${AWS_ENDPOINT_URL:-}" ]] && echo yes || echo no)
AWS_REGION: ${AWS_REGION:-unset}
AWS_DEFAULT_REGION: ${AWS_DEFAULT_REGION:-unset}
AWS_PROFILE: ${AWS_PROFILE:-unset}
AWS_ACCESS_KEY_ID set: $([[ -n "${AWS_ACCESS_KEY_ID:-}" ]] && echo yes || echo no)
AWS_SECRET_ACCESS_KEY set: $([[ -n "${AWS_SECRET_ACCESS_KEY:-}" ]] && echo yes || echo no)
AWS_SESSION_TOKEN set: $([[ -n "${AWS_SESSION_TOKEN:-}" ]] && echo yes || echo no)

Local staging file: ${LOCAL_FILE}
EOF
}

append_verification() {
    {
        echo
        echo "S3 verification"
        echo "----------------"
        echo "Uploaded fourth.txt to: ${REMOTE_URI}"
        echo
        echo "aws s3 ls output:"
        aws s3 ls "${REMOTE_URI}"
        echo
        echo "Downloaded object to: ${DOWNLOADED_FILE}"
        echo "Local sha256:"
        sha256sum "${LOCAL_FILE}"
        echo "Downloaded sha256:"
        sha256sum "${DOWNLOADED_FILE}"
    } >> "${LOCAL_FILE}"
}

write_report

echo "Uploading initial report to ${REMOTE_URI}"
aws s3 cp "${LOCAL_FILE}" "${REMOTE_URI}"

echo "Downloading report back from ${REMOTE_URI}"
aws s3 cp "${REMOTE_URI}" "${DOWNLOADED_FILE}"

append_verification

echo "Uploading final report with verification to ${REMOTE_URI}"
aws s3 cp "${LOCAL_FILE}" "${REMOTE_URI}"

echo "Wrote ${REMOTE_URI}"
