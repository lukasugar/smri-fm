#!/usr/bin/env bash
#SBATCH --job-name=smri_fm_internal_eval
#SBATCH --output=slurms/%x-%j.out
#SBATCH --error=slurms/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=training

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/internal_evals/load_env.sh
source "$REPO_ROOT/scripts/internal_evals/load_env.sh"

echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-not_set}"
echo "Working directory: $(pwd)"

uv run python -m evaluation.main --config src/evaluation/config/fomo_brain_age_gap_probe.yaml