# Running FOMO26 evals on Slurm

This is the project-specific runbook for running smri-fm finetuning/eval jobs on
the MedARC/C-Gen Nebius Slurm cluster. It combines the local Slurm guide in
`.scratch/slurm_guide.md` with the `src/asparagus_bridge` setup.

The important points are:

- Put code and small config files in `$HOME`; put datasets, checkpoints, caches,
  and run outputs under `/data/$USER`.
- Use `uv` for the Python environment.
- Use `.env` plus `source scripts/setup_asparagus_env.sh` to make Asparagus see
  the cluster paths inside Slurm jobs.
- Use R2/S3 for cold storage and for moving large artifacts on and off the
  cluster.
- Submit training/eval with `sbatch`; do not run heavy work on login nodes.

## 1. Connect and choose cluster paths

Connect to a login node:

```bash
ssh login-1.sophont-n.cgen
# or
ssh login-2.sophont-n.cgen
```

Use `tmux` so log monitoring survives disconnects:

```bash
tmux new -s smri-evals
```

Define the cluster layout for this project:

```bash
export PROJECT_ROOT="$HOME/code/smri-fm"
export DATA_ROOT="/data/$USER/smri-fm"
export R2_PREFIX="s3://medarc/$USER/smri-fm"
```

Recommended directories:

```text
$PROJECT_ROOT/                         repo clone
$DATA_ROOT/asparagus/source/           raw FOMO26 zip/extracted data
$DATA_ROOT/asparagus/data/             processed Asparagus task datasets
$DATA_ROOT/asparagus/raw_labels/       formatted raw labels
$DATA_ROOT/asparagus/models/           Hydra run dirs, checkpoints, predictions
$DATA_ROOT/asparagus/results/          optional result exports
$DATA_ROOT/checkpoints/                smri-fm pretrain checkpoints
$DATA_ROOT/hf, torch, wandb, cache      package/model/logger caches
```

Create them:

```bash
mkdir -p "$PROJECT_ROOT" "$DATA_ROOT"/{asparagus/source,asparagus/data,asparagus/raw_labels,asparagus/models,asparagus/results,checkpoints,hf,torch,wandb,cache}
```

## 2. Put the repo on the cluster

Clone or update the repository on a login node:

```bash
mkdir -p "$HOME/code"
cd "$HOME/code"
git clone <repo-url> smri-fm
cd "$PROJECT_ROOT"
git submodule update --init --recursive
```

If the repo is already present:

```bash
cd "$PROJECT_ROOT"
git pull
git submodule update --init --recursive
```

Install the Python environment in the current folder:

```bash
cd "$PROJECT_ROOT"
uv venv
uv sync
```

Do a cheap import check on the login node:

```bash
uv run python - <<'PY'
import asparagus
import asparagus_preprocessing
import asparagus_bridge
print("asparagus stack imports OK")
PY
```

## 3. Configure `.env` for cluster paths

Create a repo-root `.env` on the cluster. This file is ignored by git and is
loaded by `scripts/setup_asparagus_env.sh`.

```bash
cd "$PROJECT_ROOT"
cat > .env <<EOF
ASPARAGUS_SOURCE=$DATA_ROOT/asparagus/source
ASPARAGUS_DATA=$DATA_ROOT/asparagus/data
ASPARAGUS_RAW_LABELS=$DATA_ROOT/asparagus/raw_labels
ASPARAGUS_MODELS=$DATA_ROOT/asparagus/models
ASPARAGUS_RESULTS=$DATA_ROOT/asparagus/results

HF_HOME=$DATA_ROOT/hf
TRANSFORMERS_CACHE=$DATA_ROOT/hf
HF_DATASETS_CACHE=$DATA_ROOT/hf/datasets
TORCH_HOME=$DATA_ROOT/torch
WANDB_DIR=$DATA_ROOT/wandb
XDG_CACHE_HOME=$DATA_ROOT/cache
UV_CACHE_DIR=$DATA_ROOT/cache/uv

# Optional, if using wandb from batch jobs:
# WANDB_ENTITY=...
# WANDB_PROJECT=...
# WANDB_API_KEY=...
EOF
chmod 600 .env
```

Check that the project setup script resolves to `/data` paths:

```bash
source scripts/setup_asparagus_env.sh
```

You should see:

```text
ASPARAGUS_SOURCE=/data/<user>/smri-fm/asparagus/source
ASPARAGUS_DATA=/data/<user>/smri-fm/asparagus/data
ASPARAGUS_MODELS=/data/<user>/smri-fm/asparagus/models
...
```

For Slurm jobs, source the same script inside the `.sbatch` file after `cd
"$PROJECT_ROOT"`. Do not rely on an interactive shell having already sourced
it.

## 4. Configure R2/S3 access

Use Cloudflare R2 for moving raw data, processed data, checkpoints, and final
results. The Slurm guide uses the `medarc` bucket and the AWS CLI interface.

Set credentials in the current shell before `aws s3` commands:

```bash
export AWS_ENDPOINT_URL="https://<account-id>.r2.cloudflarestorage.com"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="auto"
```

For repeated use, put only your private credentials in a file outside the repo:

```bash
cat > "$HOME/.r2-env" <<'EOF'
export AWS_ENDPOINT_URL="https://<account-id>.r2.cloudflarestorage.com"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="auto"
EOF
chmod 600 "$HOME/.r2-env"
```

Then source it in login shells or in jobs that need to sync outputs:

```bash
source "$HOME/.r2-env"
aws s3 ls s3://medarc/
aws s3 ls "$R2_PREFIX/"
```

If `aws` is not available, install or load the AWS CLI on the cluster before the
sync steps. The project has `scripts/s3_report_gpu.sbatch` to test that R2
credentials are visible from a GPU allocation:

```bash
cd "$PROJECT_ROOT"
mkdir -p slurms
source "$HOME/.r2-env"
sbatch scripts/s3_report_gpu.sbatch "$R2_PREFIX/test-output"
```

## 5. Move data to the cluster

There are two workable patterns.

### Option A: R2 as transfer/staging

From the machine that has the data:

```bash
source /path/to/r2-env
aws s3 sync /local/path/to/FOMO26/raw "$R2_PREFIX/asparagus/source"
aws s3 cp /local/path/to/checkpoint-last.pth "$R2_PREFIX/checkpoints/checkpoint-last.pth"
```

On the cluster login node:

```bash
cd "$PROJECT_ROOT"
source scripts/setup_asparagus_env.sh
source "$HOME/.r2-env"
aws s3 sync "$R2_PREFIX/asparagus/source" "$ASPARAGUS_SOURCE"
aws s3 sync "$R2_PREFIX/asparagus/data" "$ASPARAGUS_DATA"
aws s3 cp "$R2_PREFIX/checkpoints/checkpoint-last.pth" "$DATA_ROOT/checkpoints/checkpoint-last.pth"
```

Use `--dryrun` before large or destructive syncs:

```bash
aws s3 sync "$R2_PREFIX/asparagus/data" "$ASPARAGUS_DATA" --dryrun
```

### Option B: Copy directly to `/data`

If direct SSH transfer is easier:

```bash
rsync -avP /local/path/to/FOMO26/raw/ 'login-1.sophont-n.cgen:/data/<cluster-user>/smri-fm/asparagus/source/'
rsync -avP /local/path/to/checkpoint-last.pth 'login-1.sophont-n.cgen:/data/<cluster-user>/smri-fm/checkpoints/'
```

Keep large raw/processed task folders under `/data`, not under `$HOME`.

## 6. Prepare FOMO26 task data

Asparagus finetuning uses processed task directories under
`$ASPARAGUS_DATA/<task>`. Raw FOMO26 files go in `$ASPARAGUS_SOURCE`.

First source the project environment:

```bash
cd "$PROJECT_ROOT"
source scripts/setup_asparagus_env.sh
```

Task 1, infarct classification, FLAIR-only:

```bash
cd "$ASPARAGUS_SOURCE"
unzip -n Task_1.zip -d Task_1

cd "$PROJECT_ROOT"
uv run asp_process \
  --dataset CLS002_FOMO26_Infarct_CUSTOM \
  --task_name CLS002_FOMO26_Infarct_FLAIR \
  --modalities flair \
  --save_as_tensor \
  --num_workers 16

uv run asp_split --dataset CLS002_FOMO26_Infarct_FLAIR --vals 80 10 10
```

Task 3, brain age regression:

```bash
cd "$ASPARAGUS_SOURCE"
unzip -n Task_3.zip -d Task_3

cd "$PROJECT_ROOT"
uv run asp_process --dataset REGR002 --save_as_tensor --num_workers 16
uv run asp_split --dataset REGR002_FOMO26_BrainAge --vals 80 10 10
```

Task 5, polymicrogyria classification:

```bash
# Required files:
# $ASPARAGUS_SOURCE/Task_5_extract.py
# $ASPARAGUS_SOURCE/Zhang_Lingfeng_2022_PPMR_Dataset.zip

cd "$ASPARAGUS_SOURCE"
uv run --project "$PROJECT_ROOT" python Task_5_extract.py --verbose

cd "$PROJECT_ROOT"
uv run asp_process --dataset CLS003 --save_as_tensor --num_workers 16
uv run asp_split --dataset CLS003_FOMO26_Polymicrogyria --vals 80 10 10
```

After preprocessing, verify the expected files exist:

```bash
find "$ASPARAGUS_DATA" -maxdepth 2 \( -name dataset.json -o -name paths.json -o -name 'split_80_10_10.json' -o -name 'TEST_80_10_10.json' \) -print
```

For expensive preprocessing, submit the same commands in a CPU/GPU batch job
rather than running them on the login node.

## 7. Convert the pretrain checkpoint

The bridge needs an Asparagus-compatible checkpoint. Convert once on the
cluster, using the same Python environment:

```bash
cd "$PROJECT_ROOT"
source scripts/setup_asparagus_env.sh

uv run python - <<PY
from asparagus_bridge.checkpoint import convert_checkpoint
convert_checkpoint(
    "smri_mae",
    "$DATA_ROOT/checkpoints/checkpoint-last.pth",
    "$DATA_ROOT/checkpoints/checkpoint-last.asparagus.ckpt",
)
PY
```

The converted path is the value to pass as `checkpoint_path=...` in finetuning
and linear probing commands.

## 8. Smoke test interactively on one GPU

Before launching long jobs, get an allocated shell:

```bash
srun --partition=main --account=training --qos=normal --gpus=1 --cpus-per-task=16 --mem=64G --time=01:00:00 --pty bash
```

Inside the allocation:

```bash
cd "$PROJECT_ROOT"
source scripts/setup_asparagus_env.sh
nvidia-smi

uv run asp_finetune_cls \
  task=CLS002_FOMO26_Infarct_FLAIR \
  +model=smri_mae \
  checkpoint_path="$DATA_ROOT/checkpoints/checkpoint-last.asparagus.ckpt" \
  data.train_split=split_80_10_10 \
  data.test_split=TEST_80_10_10 \
  hardware.num_workers=12 \
  training.epochs=1 \
  training.limit_train_batches=2 \
  training.limit_val_batches=1 \
  logger.wandb_logging=false
```

This should write a Hydra run directory under `$ASPARAGUS_MODELS`. Classification
and regression finetune commands run test/eval after training; predictions are
written below the run directory, for example:

```text
predictions/<task>__TEST_80_10_10__best.json
```

Exit the allocation after the smoke test:

```bash
exit
```

## 9. Submit real eval jobs with `sbatch`

Create `scripts/eval_fomo26_one_gpu.sbatch` on the cluster, or use this as the
template for a committed script later:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=smri_fomo26_eval
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurms/%x-%j.out
#SBATCH --error=slurms/%x-%j.err
#SBATCH --requeue

set -euo pipefail

cd "${PROJECT_ROOT:?Set PROJECT_ROOT before sbatch, or hard-code it here.}"
mkdir -p slurms

# Optional if the job uploads outputs to R2.
if [[ -f "$HOME/.r2-env" ]]; then
    source "$HOME/.r2-env"
fi

source scripts/setup_asparagus_env.sh

echo "Job ID: ${SLURM_JOB_ID:-unset}"
echo "Node list: ${SLURM_JOB_NODELIST:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "ASPARAGUS_DATA: ${ASPARAGUS_DATA}"
echo "ASPARAGUS_MODELS: ${ASPARAGUS_MODELS}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to converted .ckpt}"
TASK_KIND="${TASK_KIND:?Set TASK_KIND to cls, reg, or linear_probe}"
TASK_NAME="${TASK_NAME:?Set TASK_NAME to an Asparagus task name}"

COMMON_OVERRIDES=(
  "task=${TASK_NAME}"
  "+model=smri_mae"
  "checkpoint_path=${CHECKPOINT_PATH}"
  "data.train_split=${TRAIN_SPLIT:-split_80_10_10}"
  "data.test_split=${TEST_SPLIT:-TEST_80_10_10}"
  "data.fold=${FOLD:-0}"
  "hardware.num_workers=${SLURM_CPUS_PER_TASK:-16}"
  "logger.wandb_logging=${WANDB_LOGGING:-false}"
)

case "$TASK_KIND" in
  cls)
    uv run asp_finetune_cls "${COMMON_OVERRIDES[@]}"
    ;;
  reg)
    uv run asp_finetune_reg "${COMMON_OVERRIDES[@]}"
    ;;
  linear_probe)
    uv run asp_linear_probe "${COMMON_OVERRIDES[@]}"
    ;;
  *)
    echo "Unknown TASK_KIND: $TASK_KIND. Use cls, reg, or linear_probe." >&2
    exit 2
    ;;
esac

if [[ -n "${R2_RESULTS_URI:-}" ]] && command -v aws >/dev/null 2>&1; then
    aws s3 sync "$ASPARAGUS_MODELS" "$R2_RESULTS_URI/models"
fi
```

Make it executable:

```bash
chmod +x scripts/eval_fomo26_one_gpu.sbatch
```

Submit Task 1 classification:

```bash
cd "$PROJECT_ROOT"
mkdir -p slurms
export PROJECT_ROOT="$PROJECT_ROOT"
export CHECKPOINT_PATH="$DATA_ROOT/checkpoints/checkpoint-last.asparagus.ckpt"
export TASK_KIND=cls
export TASK_NAME=CLS002_FOMO26_Infarct_FLAIR
export WANDB_LOGGING=false
export R2_RESULTS_URI="$R2_PREFIX/results/task1"

sbatch --export=ALL scripts/eval_fomo26_one_gpu.sbatch
```

Submit Task 3 regression:

```bash
export TASK_KIND=reg
export TASK_NAME=REGR002_FOMO26_BrainAge
export R2_RESULTS_URI="$R2_PREFIX/results/task3"
sbatch --export=ALL scripts/eval_fomo26_one_gpu.sbatch
```

Submit Task 5 classification:

```bash
export TASK_KIND=cls
export TASK_NAME=CLS003_FOMO26_Polymicrogyria
export R2_RESULTS_URI="$R2_PREFIX/results/task5"
sbatch --export=ALL scripts/eval_fomo26_one_gpu.sbatch
```

Submit linear probing for Task 1:

```bash
export TASK_KIND=linear_probe
export TASK_NAME=CLS002_FOMO26_Infarct_FLAIR
export R2_RESULTS_URI="$R2_PREFIX/results/task1-linear-probe"
sbatch --export=ALL scripts/eval_fomo26_one_gpu.sbatch
```

If you prefer not to export variables in the login shell, hard-code the stable
paths at the top of the `.sbatch` file. Keep credentials out of the repo.

For R2 credentials in a batch job, either source `$HOME/.r2-env` inside the
script as shown above, or source it before `sbatch` and submit with
`--export=ALL`. Prefer the file approach because it does not depend on the
state of the current login shell.

## 10. Optional: submit multiple evals as an array

For the current bridge, the safest array dimension is task/job type, one GPU per
array element:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=smri_fomo26_array
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%2
#SBATCH --output=slurms/%x-%A-%a.out
#SBATCH --error=slurms/%x-%A-%a.err

set -euo pipefail

cd "${PROJECT_ROOT:?Set PROJECT_ROOT before sbatch}"
source scripts/setup_asparagus_env.sh

CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH}"

KINDS=(cls reg cls)
TASKS=(
  CLS002_FOMO26_Infarct_FLAIR
  REGR002_FOMO26_BrainAge
  CLS003_FOMO26_Polymicrogyria
)

TASK_KIND="${KINDS[$SLURM_ARRAY_TASK_ID]}"
TASK_NAME="${TASKS[$SLURM_ARRAY_TASK_ID]}"

COMMON_OVERRIDES=(
  "task=${TASK_NAME}"
  "+model=smri_mae"
  "checkpoint_path=${CHECKPOINT_PATH}"
  "data.train_split=split_80_10_10"
  "data.test_split=TEST_80_10_10"
  "hardware.num_workers=${SLURM_CPUS_PER_TASK:-16}"
  "logger.wandb_logging=${WANDB_LOGGING:-false}"
)

if [[ "$TASK_KIND" == "reg" ]]; then
    uv run asp_finetune_reg "${COMMON_OVERRIDES[@]}"
else
    uv run asp_finetune_cls "${COMMON_OVERRIDES[@]}"
fi
```

Submit:

```bash
export PROJECT_ROOT="$PROJECT_ROOT"
export CHECKPOINT_PATH="$DATA_ROOT/checkpoints/checkpoint-last.asparagus.ckpt"
export WANDB_LOGGING=false
sbatch --export=ALL scripts/eval_fomo26_array.sbatch
```

Use a concurrency cap such as `%2` to avoid occupying too many GPUs at once.

## 11. Monitor jobs and inspect outputs

Queue:

```bash
squeue -u "$USER"
squeue -j JOB_ID -o "%.18i %.8T %.30R"
```

Logs:

```bash
tail -f slurms/JOB_ID.out
tail -f slurms/JOB_ID.err
```

Accounting after completion:

```bash
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem,NodeList
seff JOB_ID
```

Find prediction files:

```bash
find "$ASPARAGUS_MODELS" -path '*/predictions/*.json' -print
```

Find best checkpoints:

```bash
find "$ASPARAGUS_MODELS" -path '*/checkpoints/*.ckpt' -print
```

If a job fails immediately with `ModuleNotFoundError`, verify the `.sbatch`
script uses `cd "$PROJECT_ROOT"` and `uv run ...`. If it cannot find data,
verify `source scripts/setup_asparagus_env.sh` printed `/data` paths and that
`$ASPARAGUS_DATA/<task>/dataset.json`, `paths.json`, and split JSONs exist.

## 12. Sync results back to R2

After jobs complete:

```bash
source "$HOME/.r2-env"
aws s3 sync "$ASPARAGUS_MODELS" "$R2_PREFIX/results/models"
aws s3 sync "$ASPARAGUS_RESULTS" "$R2_PREFIX/results/exports"
```

To bring results back to a local workstation:

```bash
source /path/to/r2-env
aws s3 sync "$R2_PREFIX/results" /local/path/to/smri-fm-results
```

Keep hot outputs on `/data` while iterating. Move old runs to R2 and delete
unneeded local copies once the R2 sync is verified.

## 13. Current project-specific caveats

- The checked-in smoke-test configs under `src/asparagus_bridge/configs` are CPU
  smoke tests with local absolute checkpoint paths. For cluster jobs, prefer CLI
  overrides for `checkpoint_path`, hardware, split names, and logging.
- `scripts/eval_fomo26.sh` converts a checkpoint and loops over
  `FOMO_CLS_TASKS`, `FOMO_REG_TASKS`, and `FOMO_SEG_TASKS`, but it is marked
  "not tested yet" in the bridge README. Use the explicit one-task Slurm
  template first.
- Task 1 should start with `CLS002_FOMO26_Infarct_FLAIR` for single-channel
  smri-fm checkpoints. Multi-modal Task 1 variants require preprocessing with a
  matching `--modalities` list and model input channel handling.
- Segmentation evals are still listed as TBD in `src/asparagus_bridge/README.md`.
  Do not assume `asp_finetune_seg` works for smri-fm until the segmentation
  bridge is tested.
- Asparagus writes Hydra outputs under `ASPARAGUS_MODELS`, not
  `ASPARAGUS_RESULTS`, for the finetune commands described here.
